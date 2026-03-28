import os
import time
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from .config import load_skab_config
from .data import NativeSKABLoader
from .model import build_skab_model
from .reasoner import SKABRuleReasoner
from .utils import (
    compute_metrics,
    create_run_dir,
    ensure_directory,
    seed_everything,
    select_threshold,
    setup_logger,
    write_json,
    write_rows_csv,
)


def prepare_dataset(config_path: str, logger) -> Dict[str, Any]:
    config, project_root = load_skab_config(config_path)
    loader = NativeSKABLoader(
        data_dir=config['paths']['data_dir'],
        feature_names=config['model']['feature_names'],
        window_size=config['model']['window_size'],
        categories=config['dataset'].get('categories'),
        include_anomaly_free=config['dataset'].get('include_anomaly_free', True),
    )
    dataset = loader.prepare_dataset(
        validation_split=config['training']['validation_split'],
        test_split=config['training']['test_split'],
    )
    dataset['config'] = config
    dataset['project_root'] = project_root
    logger.info(
        'Prepared dedicated SKAB dataset with train/val/test sequence shapes: %s / %s / %s',
        dataset['X_train_scaled'].shape,
        dataset['X_val_scaled'].shape,
        dataset['X_test_scaled'].shape,
    )
    return dataset


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def _metric_row(variant: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'variant': variant,
        'threshold': metrics.get('threshold'),
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'f1': metrics.get('f1'),
        'roc_auc': metrics.get('roc_auc'),
        'avg_precision': metrics.get('avg_precision'),
        'pr_auc': metrics.get('pr_auc'),
        'positive_rate': metrics.get('positive_rate'),
    }


def train_random_forest(dataset: Dict[str, Any], run_dir: str, logger) -> Dict[str, Any]:
    variant_dir = ensure_directory(os.path.join(run_dir, 'random_forest'))
    X_train = _flatten(dataset['X_train_scaled'])
    X_val = _flatten(dataset['X_val_scaled'])
    X_test = _flatten(dataset['X_test_scaled'])
    start = time.perf_counter()
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, dataset['y_train'])
    train_time = time.perf_counter() - start
    y_val_scores = model.predict_proba(X_val)[:, 1]
    threshold, val_threshold_metrics = select_threshold(
        dataset['y_val'],
        y_val_scores,
        metric_name=dataset['config']['evaluation']['selection_metric'],
    )
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(dataset['y_test'], y_scores, threshold=threshold)
    metrics['train_time_s'] = round(train_time, 3)
    metrics['validation_threshold_metrics'] = val_threshold_metrics
    joblib.dump(model, os.path.join(variant_dir, 'model.joblib'))
    write_json(
        os.path.join(variant_dir, 'metrics.json'),
        {'metrics': metrics, 'selected_threshold': threshold, 'validation_threshold_metrics': val_threshold_metrics},
    )
    logger.info('Dedicated SKAB Random Forest - F1 %.4f ROC-AUC %s threshold %.4f', metrics['f1'], metrics.get('roc_auc'), threshold)
    return {
        'variant': 'random_forest',
        'metrics': metrics,
        'y_scores': y_scores,
        'val_scores': y_val_scores,
        'selected_threshold': threshold,
    }


def train_neural_model(dataset: Dict[str, Any], run_dir: str, logger, epochs_override: int = None) -> Dict[str, Any]:
    variant_dir = ensure_directory(os.path.join(run_dir, 'neural'))
    tf.keras.backend.clear_session()
    config = dataset['config']
    model = build_skab_model(config)

    y_train = dataset['y_train']
    classes = np.unique(y_train)
    class_weights_values = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, class_weights_values)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(variant_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
        ),
    ]

    epochs = int(epochs_override or config['training']['epochs'])
    start = time.perf_counter()
    history = model.fit(
        dataset['X_train_scaled'],
        dataset['y_train'],
        validation_data=(dataset['X_val_scaled'], dataset['y_val']),
        epochs=epochs,
        batch_size=config['training']['batch_size'],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.perf_counter() - start

    y_val_scores = model.predict(dataset['X_val_scaled'], verbose=0).ravel()
    threshold, val_threshold_metrics = select_threshold(
        dataset['y_val'],
        y_val_scores,
        metric_name=config['evaluation']['selection_metric'],
    )
    y_test_scores = model.predict(dataset['X_test_scaled'], verbose=0).ravel()
    metrics = compute_metrics(dataset['y_test'], y_test_scores, threshold=threshold)
    metrics['train_time_s'] = round(train_time, 3)
    metrics['validation_threshold_metrics'] = val_threshold_metrics

    model.save(os.path.join(variant_dir, 'final_model.keras'))
    joblib.dump(dataset['scaler'], os.path.join(variant_dir, 'scaler.joblib'))
    write_json(
        os.path.join(variant_dir, 'metrics.json'),
        {
            'metrics': metrics,
            'history': history.history,
            'selected_threshold': threshold,
            'validation_threshold_metrics': val_threshold_metrics,
        },
    )
    logger.info('Dedicated SKAB neural model - F1 %.4f ROC-AUC %s threshold %.4f', metrics['f1'], metrics.get('roc_auc'), threshold)
    return {
        'variant': 'neural',
        'metrics': metrics,
        'y_scores': y_test_scores,
        'val_scores': y_val_scores,
        'selected_threshold': threshold,
    }


def train_symbolic_model(dataset: Dict[str, Any], run_dir: str, logger) -> Dict[str, Any]:
    variant_dir = ensure_directory(os.path.join(run_dir, 'symbolic'))
    symbolic_cfg = dataset['config']['symbolic']
    reasoner = SKABRuleReasoner(
        max_rules=symbolic_cfg['max_rules'],
        min_rule_precision=symbolic_cfg['min_rule_precision'],
        min_rule_recall=symbolic_cfg['min_rule_recall'],
        selection_metric=dataset['config']['evaluation']['selection_metric'],
    )
    fit_summary = reasoner.fit(
        dataset['X_train_raw'],
        dataset['y_train'],
        dataset['X_val_raw'],
        dataset['y_val'],
        dataset['feature_names'],
    )
    val_scores = reasoner.predict_scores(dataset['X_val_raw'], dataset['feature_names'])
    evaluation = reasoner.evaluate(dataset['X_test_raw'], dataset['y_test'], dataset['feature_names'])
    explanations = reasoner.explain_samples(dataset['X_test_raw'], dataset['feature_names'])
    write_json(
        os.path.join(variant_dir, 'metrics.json'),
        {
            'fit_summary': fit_summary,
            'evaluation': evaluation,
            'sample_explanations': explanations,
        },
    )
    write_rows_csv(
        os.path.join(variant_dir, 'rules.csv'),
        [
            {
                'name': rule['name'],
                'feature': rule['feature'],
                'direction': rule['direction'],
                'threshold': rule['threshold'],
                'precision': rule['precision'],
                'recall': rule['recall'],
                'f1': rule['f1'],
                'weight': rule['weight'],
                'support': rule['support'],
            }
            for rule in evaluation['rules']
        ],
    )
    logger.info('Dedicated SKAB symbolic model - F1 %.4f ROC-AUC %s', evaluation['metrics']['f1'], evaluation['metrics'].get('roc_auc'))
    return {
        'variant': 'symbolic',
        'metrics': evaluation['metrics'],
        'y_scores': evaluation['scores'],
        'val_scores': val_scores,
        'rules': evaluation['rules'],
        'score_threshold': evaluation['score_threshold'],
    }


def combine_neuro_symbolic(dataset: Dict[str, Any], run_dir: str, logger, neural_result: Dict[str, Any], symbolic_result: Dict[str, Any]) -> Dict[str, Any]:
    variant_dir = ensure_directory(os.path.join(run_dir, 'neuro_symbolic'))
    best = None
    for alpha in np.linspace(0.1, 0.9, 9):
        val_combo = alpha * neural_result['val_scores'] + (1.0 - alpha) * symbolic_result.get('val_scores', np.zeros_like(neural_result['val_scores']))
        threshold, val_metrics = select_threshold(dataset['y_val'], val_combo, metric_name=dataset['config']['evaluation']['selection_metric'])
        if best is None or val_metrics[dataset['config']['evaluation']['selection_metric']] > best['val_metric']:
            best = {
                'alpha': float(alpha),
                'threshold': float(threshold),
                'val_metric': float(val_metrics[dataset['config']['evaluation']['selection_metric']]),
                'val_metrics': val_metrics,
            }

    y_scores = best['alpha'] * neural_result['y_scores'] + (1.0 - best['alpha']) * symbolic_result['y_scores']
    metrics = compute_metrics(dataset['y_test'], y_scores, threshold=best['threshold'])
    write_json(
        os.path.join(variant_dir, 'metrics.json'),
        {
            'metrics': metrics,
            'selected_alpha': best['alpha'],
            'selected_threshold': best['threshold'],
            'validation_metrics': best['val_metrics'],
        },
    )
    logger.info('Dedicated SKAB neuro-symbolic model - F1 %.4f ROC-AUC %s alpha %.2f', metrics['f1'], metrics.get('roc_auc'), best['alpha'])
    return {
        'variant': 'neuro_symbolic',
        'metrics': metrics,
        'y_scores': y_scores,
        'selected_alpha': best['alpha'],
        'selected_threshold': best['threshold'],
    }


def run_pipeline(config_path: str, run_name: str = None, modes: Sequence[str] = ('random_forest', 'neural', 'symbolic', 'neuro_symbolic'), seed: int = 42, epochs: int = None) -> Dict[str, Any]:
    config, _ = load_skab_config(config_path)
    run_dir = create_run_dir(config['paths']['results_dir'], run_name=run_name)
    logger = setup_logger('ANSRDT.SKAB.Separate', os.path.join(run_dir, 'run.log'))
    seed_everything(seed)
    dataset = prepare_dataset(config_path, logger)

    results: Dict[str, Any] = {
        'run_dir': run_dir,
        'dataset_summary': {
            'train_shape': dataset['X_train_scaled'].shape,
            'val_shape': dataset['X_val_scaled'].shape,
            'test_shape': dataset['X_test_scaled'].shape,
            'train_anomaly_rate': float(np.mean(dataset['y_train'])),
            'val_anomaly_rate': float(np.mean(dataset['y_val'])),
            'test_anomaly_rate': float(np.mean(dataset['y_test'])),
            'feature_names': dataset['feature_names'],
        },
        'results': {},
    }

    if 'random_forest' in modes:
        results['results']['random_forest'] = train_random_forest(dataset, run_dir, logger)
    if 'neural' in modes or 'neuro_symbolic' in modes:
        results['results']['neural'] = train_neural_model(dataset, run_dir, logger, epochs_override=epochs)
    if 'symbolic' in modes or 'neuro_symbolic' in modes:
        results['results']['symbolic'] = train_symbolic_model(dataset, run_dir, logger)
    if 'neuro_symbolic' in modes:
        results['results']['neuro_symbolic'] = combine_neuro_symbolic(
            dataset,
            run_dir,
            logger,
            results['results']['neural'],
            results['results']['symbolic'],
        )

    summary_rows: List[Dict[str, Any]] = []
    for variant_name, variant_result in results['results'].items():
        if 'metrics' in variant_result:
            row = _metric_row(variant_name, variant_result['metrics'])
            if 'selected_alpha' in variant_result:
                row['selected_alpha'] = variant_result['selected_alpha']
            if 'selected_threshold' in variant_result:
                row['selected_threshold'] = variant_result['selected_threshold']
            summary_rows.append(row)

    write_rows_csv(os.path.join(run_dir, 'summary.csv'), summary_rows)
    write_json(os.path.join(run_dir, 'run_summary.json'), results)
    return results
