# src/skab/config.py
# Loads dedicated SKAB configuration, validates required sections, and resolves project-relative paths so benchmark runs remain portable across entrypoints and environments.

import os
from typing import Any, Dict, Tuple

import yaml


DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def load_skab_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    # Resolve project-local paths relative to the config file so the dedicated
    # SKAB runner can be launched from the repo root or from module form.
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as handle:
        config = yaml.safe_load(handle)

    for section in ['dataset', 'model', 'training', 'paths']:
        if section not in config:
            raise KeyError(f"Missing required configuration section: {section}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    paths = config.setdefault('paths', {})
    for key, value in list(paths.items()):
        if isinstance(value, str) and not os.path.isabs(value):
            paths[key] = os.path.normpath(os.path.join(project_root, value))

    return config, project_root
