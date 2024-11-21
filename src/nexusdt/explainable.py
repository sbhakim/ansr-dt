# src/nexusdt/explainable.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import json
import os
from datetime import datetime
from .core import NEXUSDTCore

from src.reasoning.reasoning import SymbolicReasoner
from src.reasoning.knowledge_graph import KnowledgeGraphGenerator


class ExplainableNEXUSDT(NEXUSDTCore):
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None,
                 cnn_lstm_model: Optional[Any] = None, ppo_agent: Optional[Any] = None,
                 symbolic_reasoner: Optional[SymbolicReasoner] = None,
                 knowledge_graph: Optional[KnowledgeGraphGenerator] = None):
        super().__init__(config_path, logger, cnn_lstm_model, ppo_agent)
        self.symbolic_reasoner = symbolic_reasoner
        self.knowledge_graph = knowledge_graph
        self.decision_history = []
        self.explanation_templates = {
            'normal': "System is operating within normal parameters.",
            'anomaly': "Anomaly detected with {confidence:.2%} confidence.",
            'action': "Recommended adjustments: {action}",
            'insights': "System insights: {insights}",
            'critical': "CRITICAL: Immediate attention required! {reasons}"
        }

    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Generate detailed human-readable explanation."""
        try:
            if not decision['action']:
                return self.explanation_templates['normal']

            explanations = []

            # Add anomaly explanation
            explanations.append(
                self.explanation_templates['anomaly'].format(
                    confidence=decision['confidence']
                )
            )

            # Add insights if available
            if decision.get('insights'):
                explanations.append(
                    self.explanation_templates['insights'].format(
                        insights=', '.join(decision['insights'])
                    )
                )

            # Add action explanation
            action_str = self._format_action(decision['action'])
            explanations.append(
                self.explanation_templates['action'].format(action=action_str)
            )

            # Add critical warning if necessary
            if decision['confidence'] > 0.8 or any('Critical' in i for i in decision.get('insights', [])):
                explanations.append(
                    self.explanation_templates['critical'].format(
                        reasons=self._get_critical_reasons(decision)
                    )
                )

            return " | ".join(explanations)

        except Exception as e:
            self.logger.error(f"Error generating explanation: {e}")
            return "Error generating explanation"

    def _format_action(self, action: List[float]) -> str:
        """Format action values for human reading."""
        return (
            f"Temperature ({action[0]:.2f}), "
            f"Vibration ({action[1]:.2f}), "
            f"Pressure ({action[2]:.2f})"
        )

    def _get_critical_reasons(self, decision: Dict[str, Any]) -> str:
        """Get reasons for critical status."""
        reasons = []
        if decision['confidence'] > 0.8:
            reasons.append(f"High anomaly confidence ({decision['confidence']:.2%})")
        if decision.get('insights'):
            critical_insights = [i for i in decision['insights'] if 'Critical' in i]
            reasons.extend(critical_insights)
        return ', '.join(reasons)

    def track_decision(self, decision: Dict[str, Any]):
        """Track decision with explanation."""
        try:
            tracked_decision = {
                'timestamp': datetime.now().isoformat(),
                'decision': decision,
                'explanation': self.explain_decision(decision)
            }
            self.decision_history.append(tracked_decision)

            # Limit history size
            if len(self.decision_history) > 1000:
                self.decision_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error tracking decision: {e}")

    def get_decision_summary(self) -> pd.DataFrame:
        """Get summary of decision history."""
        try:
            if not self.decision_history:
                return pd.DataFrame()

            df = pd.DataFrame(self.decision_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['has_action'] = df['decision'].apply(
                lambda x: x.get('action') is not None
            )
            df['confidence'] = df['decision'].apply(
                lambda x: x.get('confidence', 0)
            )

            return df

        except Exception as e:
            self.logger.error(f"Error generating decision summary: {e}")
            return pd.DataFrame()

    def save_decision_history(self, output_path: str):
        """Save decision history to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    'decisions': self.decision_history,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            self.logger.info(f"Decision history saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving decision history: {e}")

    def load_decision_history(self, input_path: str):
        """Load decision history from file."""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            self.decision_history = data['decisions']
            self.logger.info(f"Decision history loaded from {input_path}")

        except Exception as e:
            self.logger.error(f"Error loading decision history: {e}")

    def get_neurosymbolic_explanation(self, decision: Dict[str, Any]) -> str:
        """Generate explanation combining neural and symbolic insights."""
        explanations = []

        # Add neural model confidence
        if decision['confidence'] > 0.5:
            explanations.append(
                f"Neural model detected anomaly with {decision['confidence']:.2%} confidence"
            )

        # Add learned rule insights
        if decision.get('learned_rules'):
            explanations.append(
                "Learned patterns: " +
                ", ".join([f"{rule} (conf: {conf:.2f})"
                           for rule, conf in decision['learned_rules']])
            )

        # Add symbolic insights
        if decision.get('insights'):
            explanations.append(
                "Symbolic insights: " + ", ".join(decision['insights'])
            )

        return " | ".join(explanations)

    def track_neurosymbolic_decisions(self, decision: Dict[str, Any]):
        """Track decisions with neural and symbolic components."""
        try:
            tracked_decision = {
                'timestamp': datetime.now().isoformat(),
                'neural_confidence': decision['confidence'],
                'learned_rules': decision.get('learned_rules', []),
                'symbolic_insights': decision.get('insights', []),
                'explanation': self.get_neurosymbolic_explanation(decision)
            }
            self.decision_history.append(tracked_decision)

            # Save periodically
            if len(self.decision_history) % 100 == 0:
                self.save_decision_history('results/neurosymbolic_decisions.json')

        except Exception as e:
            self.logger.error(f"Error tracking neurosymbolic decision: {e}")
