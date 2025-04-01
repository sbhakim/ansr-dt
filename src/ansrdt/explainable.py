# src/ansrdt/explainable.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import json
import os
from datetime import datetime
import math
from .core import ANSRDTCore


class ExplainableANSRDT(ANSRDTCore):
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None,
                 cnn_lstm_model: Optional[Any] = None, ppo_agent: Optional[Any] = None):
        """Initialize explainable ANSR-DT."""
        super().__init__(config_path, logger, cnn_lstm_model, ppo_agent)
        self.decision_history = []
        self.explanation_templates = {
            'normal': "System is operating within normal parameters.",
            'anomaly': "Anomaly detected with {confidence:.2%} confidence ({confidence_percent:.1f}%).",
            'action': "Recommended adjustments: {action}",
            'insights': "System insights: {insights}",
            'critical': "CRITICAL: Immediate attention required! {reasons}"
        }

    def explain_decision(self, decision: Dict[str, Any]) -> str:
        """Generate detailed human-readable explanation with multiple levels of detail."""
        try:
            # If no action, it's normal operation
            if not decision['action']:
                return self.explanation_templates['normal']

            explanations = []

            # Add anomaly explanation with percentage
            confidence_percent = decision['confidence'] * 100
            explanations.append(
                self.explanation_templates['anomaly'].format(
                    confidence=decision['confidence'],
                    confidence_percent=confidence_percent
                )
            )

            # Add detailed insights with categorization
            if decision.get('insights'):
                # Group insights by type
                grouped_insights = self._categorize_insights(decision['insights'])

                for category, insights in grouped_insights.items():
                    if insights:
                        explanations.append(f"{category}: {', '.join(insights)}")

            # Add action explanation with more details
            if decision.get('action'):
                action_str = self._format_action(decision['action'])
                explanations.append(
                    self.explanation_templates['action'].format(action=action_str)
                )
                # Add expected impact of actions
                impact = self._predict_action_impact(decision['action'], decision.get('current_state', {}))
                if impact:
                    explanations.append(f"Expected impact: {impact}")

            # Add critical warning if necessary with more details
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

    def _categorize_insights(self, insights: List[str]) -> Dict[str, List[str]]:
        """Categorize insights into meaningful groups."""
        categories = {
            "State observations": [],
            "Sensor anomalies": [],
            "Performance issues": [],
            "Maintenance alerts": []
        }

        for insight in insights:
            insight_lower = insight.lower()
            if any(term in insight_lower for term in ['state', 'degraded', 'critical']):
                categories["State observations"].append(insight)
            elif any(term in insight_lower for term in ['temperature', 'vibration', 'pressure']):
                categories["Sensor anomalies"].append(insight)
            elif any(term in insight_lower for term in ['efficiency', 'performance']):
                categories["Performance issues"].append(insight)
            elif any(term in insight_lower for term in ['maintenance', 'operational hours']):
                categories["Maintenance alerts"].append(insight)
            else:
                # Fallback for uncategorized insights
                for category in categories:
                    if not categories[category]:
                        categories[category].append(insight)
                        break

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _predict_action_impact(self, action: List[float], current_state: Dict[str, Any]) -> str:
        """Predict the impact of the recommended actions."""
        impacts = []

        if abs(action[0]) > 1.0:  # Temperature adjustment
            direction = "decrease" if action[0] < 0 else "increase"
            impacts.append(f"Temperature {direction} should reduce thermal stress")

        if abs(action[1]) > 1.0:  # Vibration adjustment
            direction = "decrease" if action[1] < 0 else "increase"
            impacts.append(f"Vibration {direction} should stabilize mechanical components")

        if abs(action[2]) > 1.0:  # Pressure adjustment
            direction = "decrease" if action[2] < 0 else "increase"
            impacts.append(f"Pressure {direction} should optimize fluid dynamics")

        return "; ".join(impacts) if impacts else "Minor adjustments with limited impact"

    def _format_action(self, action: List[float]) -> str:
        """Format action values for human reading."""
        return (
            f"Temperature ({action[0]:.2f}), "
            f"Vibration ({action[1]:.2f}), "
            f"Pressure ({action[2]:.2f})"
        )

    def _get_critical_reasons(self, decision: Dict[str, Any]) -> str:
        """Get reasons for critical status with enhanced detail."""
        reasons = []

        # Check confidence level and categorize severity
        confidence = decision['confidence']
        if confidence > 0.9:
            reasons.append(f"Very high anomaly confidence ({confidence:.2%})")
        elif confidence > 0.8:
            reasons.append(f"High anomaly confidence ({confidence:.2%})")

        # Check sensor values if available
        current_state = decision.get('current_state', {})
        sensor_readings = current_state.get('sensor_readings', {})

        if sensor_readings:
            # Check temperature
            temp = sensor_readings.get('temperature', 0.0)
            if temp > 85.0:
                reasons.append(f"Critical temperature level: {temp:.1f}")

            # Check vibration
            vib = sensor_readings.get('vibration', 0.0)
            if vib > 60.0:
                reasons.append(f"Critical vibration level: {vib:.1f}")

            # Check pressure
            press = sensor_readings.get('pressure', 0.0)
            if press < 15.0:
                reasons.append(f"Critical low pressure: {press:.1f}")

            # Check efficiency
            efficiency = sensor_readings.get('efficiency_index', 1.0)
            if efficiency < 0.5:
                reasons.append(f"Severe efficiency drop: {efficiency:.2f}")

        # Add critical insights
        if decision.get('insights'):
            critical_insights = [i for i in decision['insights'] if 'Critical' in i]
            reasons.extend(critical_insights)

        return ', '.join(reasons) if reasons else "Multiple anomaly indicators present"

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

        # Add neural model confidence with percentage
        confidence_percent = decision['confidence'] * 100 if 'confidence' in decision else 0.0
        if decision.get('confidence', 0.0) > 0.5:
            explanations.append(
                f"Neural model detected anomaly with {decision['confidence']:.2%} confidence ({confidence_percent:.1f}%)"
            )
        else:
            explanations.append(
                f"Neural model confidence: {decision.get('confidence', 0.0):.2%} ({confidence_percent:.1f}%)"
            )

        # Add learned rule insights with categorization
        if decision.get('learned_rules'):
            # Group rules by confidence level
            high_conf_rules = []
            med_conf_rules = []

            for rule, conf in decision['learned_rules']:
                if conf > 0.8:
                    high_conf_rules.append(f"{rule} (conf: {conf:.2f})")
                else:
                    med_conf_rules.append(f"{rule} (conf: {conf:.2f})")

            if high_conf_rules:
                explanations.append(
                    "High confidence patterns: " + ", ".join(high_conf_rules)
                )
            if med_conf_rules:
                explanations.append(
                    "Medium confidence patterns: " + ", ".join(med_conf_rules)
                )

        # Add symbolic insights with categorization
        if decision.get('insights'):
            grouped_insights = self._categorize_insights(decision['insights'])
            for category, insights in grouped_insights.items():
                explanations.append(
                    f"{category}: {', '.join(insights)}"
                )

        # Add action recommendations if available
        if decision.get('action'):
            action_str = self._format_action(decision['action'])
            explanations.append(f"Recommended actions: {action_str}")

            # Add expected impact
            impact = self._predict_action_impact(decision['action'], decision.get('current_state', {}))
            if impact:
                explanations.append(f"Expected impact: {impact}")

        return " | ".join(explanations)

    def track_neurosymbolic_decisions(self, decision: Dict[str, Any]):
        """Track decisions with neural and symbolic components."""
        try:
            tracked_decision = {
                'timestamp': datetime.now().isoformat(),
                'neural_confidence': decision.get('confidence', 0.0),
                'learned_rules': decision.get('learned_rules', []),
                'symbolic_insights': decision.get('insights', []),
                'explanation': self.get_neurosymbolic_explanation(decision),
                'action': decision.get('action'),
                'state': decision.get('current_state', {})
            }
            self.decision_history.append(tracked_decision)

            # Save periodically
            if len(self.decision_history) % 100 == 0:
                self.save_decision_history('results/neurosymbolic_decisions.json')

        except Exception as e:
            self.logger.error(f"Error tracking neurosymbolic decision: {e}")

    def adapt_and_explain(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate actions and explanations based on sensor data with enhanced explanations.

        Args:
            sensor_data (np.ndarray): Input sensor data.

        Returns:
            Dict[str, Any]: Decision with action, explanation, and additional metadata.
        """
        try:
            # Update state
            state = self.update_state(sensor_data)

            # Extract rules from current prediction if it's a strong anomaly
            if state['anomaly_score'] > 0.5 and self.reasoner:
                new_rules = self.reasoner.extract_rules_from_neural_model(
                    model=self.cnn_lstm,
                    input_data=sensor_data,
                    feature_names=self.feature_names,
                    threshold=0.7
                )
                if new_rules:
                    self.reasoner.update_rules(new_rules, min_confidence=0.7)
                    self.logger.info(f"Extracted {len(new_rules)} new rules from current prediction")

            # Generate decision
            decision = {
                'timestamp': state['timestamp'],
                'action': None,
                'explanation': 'Normal operation',
                'confidence': state['anomaly_score'],
                'current_state': state,  # Include full state for enhanced explanations
            }

            # Check for anomalies
            if state['anomaly_score'] > 0.5:
                decision.update({
                    'action': state['recommended_action'],
                    'explanation': self._generate_explanation(state),
                    'insights': state['insights']
                })

                # Track decision with detailed explanation
                self.track_neurosymbolic_decisions(decision)
            else:
                # Still track normal operations for continuous monitoring
                self.track_decision(decision)

            return decision

        except Exception as e:
            self.logger.error(f"Error in adapt_and_explain: {e}")
            raise

    def _generate_explanation(self, state: Dict[str, Any]) -> str:
        """
        Generate detailed explanation of system state and actions with enhanced context.

        Args:
            state (Dict[str, Any]): Current state information.

        Returns:
            str: Generated explanation string with multiple information levels.
        """
        explanation = []

        # Add anomaly detection explanation with percentage
        confidence_percent = state['anomaly_score'] * 100
        explanation.append(
            f"Anomaly detected with {state['anomaly_score']:.2%} confidence ({confidence_percent:.1f}%).")

        # Add symbolic insights with categorization if available
        if state['insights']:
            grouped_insights = self._categorize_insights(state['insights'])
            for category, insights in grouped_insights.items():
                explanation.append(f"{category}: {', '.join(insights)}")
        else:
            explanation.append("No specific symbolic insights available.")

        # Add recommended actions with detailed format
        action = state['recommended_action']
        explanation.append(
            f"Recommended adjustments: Temperature ({action[0]:.2f}), "
            f"Vibration ({action[1]:.2f}), Pressure ({action[2]:.2f})."
        )

        # Add expected impact of actions
        impact = self._predict_action_impact(action, state.get('sensor_readings', {}))
        if impact:
            explanation.append(f"Expected impact: {impact}")

        # Add temporal context if state history exists
        if len(self.state_history) > 1:
            prev_score = self.state_history[-2].get('anomaly_score', 0.0)
            score_change = state['anomaly_score'] - prev_score
            if abs(score_change) > 0.1:
                direction = "increasing" if score_change > 0 else "decreasing"
                explanation.append(f"Anomaly trend: {direction} ({abs(score_change):.2%} change)")

        return " | ".join(explanation)