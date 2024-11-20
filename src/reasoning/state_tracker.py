# src/reasoning/state_tracker.py

import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime



class StateTracker:
    def __init__(self, window_size: int = 1000):
        """Initialize enhanced state tracker."""
        self.window_size = window_size
        self.state_transitions = np.zeros((3, 3))  # 3 states: normal, degraded, critical
        self.state_history = []
        self.performance_history = []
        self.anomaly_history = []
        self.logger = logging.getLogger(__name__)

        # Add correlation tracking
        self.sensor_correlations = {
            'temp_vib': [],
            'temp_press': [],
            'vib_press': []
        }

        # Performance thresholds based on data statistics
        self.thresholds = {
            'efficiency_drop': 0.1,
            'performance_drop': 10.0,
            'rapid_state_change': 2,  # Max acceptable state changes in window
            'correlation_threshold': 0.8,
            'temperature_change': 5.0,
            'vibration_change': 3.0,
            'pressure_change': 2.0
        }



    def update(self, current_state: Dict[str, float]) -> Dict[str, Any]:
        """Update state tracking with enhanced monitoring."""
        try:
            # Extract current values with safety checks
            system_state = int(current_state.get('system_state', 0))
            sensor_readings = {
                'temperature': float(current_state.get('temperature', 0.0)),
                'vibration': float(current_state.get('vibration', 0.0)),
                'pressure': float(current_state.get('pressure', 0.0))
            }
            performance_metrics = {
                'efficiency_index': float(current_state.get('efficiency_index', 0.0)),
                'performance_score': float(current_state.get('performance_score', 0.0))
            }

            # Update transition matrix
            if self.state_history:
                prev_state = self.state_history[-1].get('system_state', 0)
                self.state_transitions[prev_state][system_state] += 1

            # Update histories with validated data
            state_record = {
                'system_state': system_state,
                'sensor_readings': sensor_readings,
                'performance_metrics': performance_metrics,
                'timestamp': str(np.datetime64('now'))
            }

            self.state_history.append(state_record)
            self.performance_history.append(performance_metrics)

            # Maintain window size
            if len(self.state_history) > self.window_size:
                self.state_history.pop(0)
                self.performance_history.pop(0)

            # Return current state information
            return {
                'current_state': system_state,
                'sensor_readings': sensor_readings,
                'performance_metrics': performance_metrics,
                'transition_matrix': self.state_transitions.tolist()
            }

        except Exception as e:
            self.logger.error(f"Error in state tracking: {str(e)}")
            return {
                'current_state': 0,
                'sensor_readings': {'temperature': 0.0, 'vibration': 0.0, 'pressure': 0.0},
                'performance_metrics': {'efficiency_index': 0.0, 'performance_score': 0.0},
                'transition_matrix': self.state_transitions.tolist()
            }

    def _update_histories(self, system_state: int,
                          sensor_readings: Dict[str, float],
                          performance_metrics: Dict[str, float]):
        """Update history arrays with length management."""
        state_record = {
            'system_state': system_state,
            'sensor_readings': sensor_readings,
            'performance_metrics': performance_metrics,
            'timestamp': str(np.datetime64('now'))
        }

        self.state_history.append(state_record)
        self.performance_history.append(performance_metrics)

        # Maintain window size
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
            self.performance_history.pop(0)

    def _update_correlations(self, sensor_readings: Dict[str, float]):
        """Update sensor correlations."""
        if len(self.state_history) > 1:
            prev_readings = self.state_history[-2]['sensor_readings']

            # Calculate correlations
            self.sensor_correlations['temp_vib'].append(
                (sensor_readings['temperature'] - prev_readings['temperature']) *
                (sensor_readings['vibration'] - prev_readings['vibration'])
            )
            self.sensor_correlations['temp_press'].append(
                (sensor_readings['temperature'] - prev_readings['temperature']) *
                (sensor_readings['pressure'] - prev_readings['pressure'])
            )
            self.sensor_correlations['vib_press'].append(
                (sensor_readings['vibration'] - prev_readings['vibration']) *
                (sensor_readings['pressure'] - prev_readings['pressure'])
            )

            # Maintain window size
            for key in self.sensor_correlations:
                if len(self.sensor_correlations[key]) > self.window_size:
                    self.sensor_correlations[key].pop(0)

    def _generate_insights(self, system_state: int,
                           sensor_readings: Dict[str, float],
                           performance_metrics: Dict[str, float]) -> List[str]:
        """Generate insights based on current state and history."""
        insights = []

        # Check state transitions
        if len(self.state_history) > 1:
            prev_state = self.state_history[-2]['system_state']
            if system_state > prev_state:
                insights.append(f"State degradation detected: {prev_state} -> {system_state}")
                if system_state == 2:  # Critical state
                    insights.append("WARNING: System entered critical state")
            elif system_state < prev_state:
                insights.append(f"State improvement detected: {prev_state} -> {system_state}")

        # Check performance trends
        if len(self.performance_history) > 1:
            prev_performance = self.performance_history[-2]

            # Check efficiency changes
            efficiency_drop = (performance_metrics['efficiency_index'] -
                               prev_performance['efficiency_index'])
            if abs(efficiency_drop) > self.thresholds['efficiency_drop']:
                insights.append(
                    f"Significant efficiency change: {efficiency_drop:.2f}")

            # Check performance score changes
            perf_drop = (performance_metrics['performance_score'] -
                         prev_performance['performance_score'])
            if abs(perf_drop) > self.thresholds['performance_drop']:
                insights.append(
                    f"Significant performance change: {perf_drop:.2f}")

        # Check sensor correlations and patterns
        correlation_insights = self._analyze_correlations()
        insights.extend(correlation_insights)

        # Add rapid change detection
        if len(self.state_history) > 1:
            prev_readings = self.state_history[-2]['sensor_readings']

            temp_change = abs(sensor_readings['temperature'] - prev_readings['temperature'])
            if temp_change > self.thresholds['temperature_change']:
                insights.append(f"Rapid temperature change: {temp_change:.2f}")

            vib_change = abs(sensor_readings['vibration'] - prev_readings['vibration'])
            if vib_change > self.thresholds['vibration_change']:
                insights.append(f"Rapid vibration change: {vib_change:.2f}")

            press_change = abs(sensor_readings['pressure'] - prev_readings['pressure'])
            if press_change > self.thresholds['pressure_change']:
                insights.append(f"Rapid pressure change: {press_change:.2f}")

        return insights

    def _analyze_correlations(self) -> List[str]:
        """Analyze sensor correlations for insights."""
        insights = []
        min_history = 10  # Minimum number of points for correlation analysis

        if len(self.sensor_correlations['temp_vib']) > min_history:
            # Temperature-Vibration correlation
            recent_temp_vib = np.mean(self.sensor_correlations['temp_vib'][-min_history:])
            if abs(recent_temp_vib) > self.thresholds['correlation_threshold']:
                insights.append(
                    f"Strong temperature-vibration correlation detected: {recent_temp_vib:.2f}")

            # Temperature-Pressure correlation
            recent_temp_press = np.mean(self.sensor_correlations['temp_press'][-min_history:])
            if abs(recent_temp_press) > self.thresholds['correlation_threshold']:
                insights.append(
                    f"Strong temperature-pressure correlation detected: {recent_temp_press:.2f}")

            # Vibration-Pressure correlation
            recent_vib_press = np.mean(self.sensor_correlations['vib_press'][-min_history:])
            if abs(recent_vib_press) > self.thresholds['correlation_threshold']:
                insights.append(
                    f"Strong vibration-pressure correlation detected: {recent_vib_press:.2f}")

        return insights

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over the history window."""
        if len(self.performance_history) < 2:
            return {}

        # Calculate trends
        efficiency_values = [p['efficiency_index'] for p in self.performance_history]
        performance_values = [p['performance_score'] for p in self.performance_history]

        try:
            # Calculate linear trends
            efficiency_trend = np.polyfit(range(len(efficiency_values)),
                                          efficiency_values, 1)[0]
            performance_trend = np.polyfit(range(len(performance_values)),
                                           performance_values, 1)[0]

            # Calculate stability metrics
            efficiency_stability = np.std(efficiency_values[-10:] if len(efficiency_values) > 10
                                          else efficiency_values)
            performance_stability = np.std(performance_values[-10:] if len(performance_values) > 10
                                           else performance_values)

            return {
                'efficiency_trend': float(efficiency_trend),
                'performance_trend': float(performance_trend),
                'efficiency_stability': float(efficiency_stability),
                'performance_stability': float(performance_stability),
                'trend_window': len(efficiency_values)
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance trends: {e}")
            return {}

    def _analyze_anomaly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in anomaly occurrence and system state changes."""
        try:
            if len(self.state_history) < 2:
                return {}

            # Get recent states
            recent_states = [record['system_state'] for record in self.state_history[-50:]]

            # Calculate state change frequency
            state_changes = sum(1 for i in range(1, len(recent_states))
                                if recent_states[i] != recent_states[i - 1])

            # Analyze degradation patterns
            degradation_count = sum(1 for i in range(1, len(recent_states))
                                    if recent_states[i] > recent_states[i - 1])

            # Analyze recovery patterns
            recovery_count = sum(1 for i in range(1, len(recent_states))
                                 if recent_states[i] < recent_states[i - 1])

            return {
                'state_change_frequency': state_changes,
                'degradation_count': degradation_count,
                'recovery_count': recovery_count,
                'current_stability': self._assess_state_stability(),
                'analysis_window': len(recent_states)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing anomaly patterns: {e}")
            return {}

    def _check_anomalies(self, system_state: int,
                         sensor_readings: Dict[str, float],
                         performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check for anomalous conditions."""
        try:
            anomalies = {
                'critical_state': system_state == 2,
                'sensor_anomalies': [],
                'performance_anomalies': [],
                'correlation_anomalies': [],
                'severity': 0
            }

            # Check sensor thresholds
            if sensor_readings['temperature'] > 80:
                anomalies['sensor_anomalies'].append('high_temperature')
            if sensor_readings['vibration'] > 55:
                anomalies['sensor_anomalies'].append('high_vibration')
            if sensor_readings['pressure'] < 20:
                anomalies['sensor_anomalies'].append('low_pressure')

            # Check performance metrics
            if performance_metrics['efficiency_index'] < 0.6:
                anomalies['performance_anomalies'].append('low_efficiency')
            if performance_metrics['performance_score'] < 60:
                anomalies['performance_anomalies'].append('low_performance')

            # Calculate anomaly severity
            anomalies['severity'] = (len(anomalies['sensor_anomalies']) +
                                     len(anomalies['performance_anomalies']) +
                                     (2 if system_state == 2 else 0))

            return anomalies

        except Exception as e:
            self.logger.error(f"Error checking anomalies: {e}")
            return {'severity': 0, 'error': str(e)}

    def _assess_state_stability(self) -> Dict[str, Any]:
        """Assess the stability of the current system state."""
        try:
            if len(self.state_history) < 10:
                return {'stability': 'unknown', 'confidence': 0.0}

            recent_states = [record['system_state'] for record in self.state_history[-10:]]
            state_changes = sum(1 for i in range(1, len(recent_states))
                                if recent_states[i] != recent_states[i - 1])

            stability_score = 1.0 - (state_changes / 9)  # 9 possible changes in 10 states

            # Classify stability
            if stability_score > 0.8:
                stability = 'stable'
            elif stability_score > 0.5:
                stability = 'moderately_stable'
            else:
                stability = 'unstable'

            return {
                'stability': stability,
                'score': float(stability_score),
                'recent_changes': state_changes,
                'window_size': len(recent_states)
            }

        except Exception as e:
            self.logger.error(f"Error assessing state stability: {e}")
            return {'stability': 'unknown', 'error': str(e)}

    def _get_current_correlations(self) -> Dict[str, float]:
        """Get current correlation values."""
        try:
            if not all(len(corr) > 0 for corr in self.sensor_correlations.values()):
                return {}

            return {
                key: float(np.mean(values[-10:]) if len(values) >= 10 else np.mean(values))
                for key, values in self.sensor_correlations.items()
            }

        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return {}

    def get_transition_probabilities(self) -> np.ndarray:
        """Calculate state transition probabilities."""
        try:
            row_sums = self.state_transitions.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            return self.state_transitions / row_sums

        except Exception as e:
            self.logger.error(f"Error calculating transition probabilities: {e}")
            return np.zeros((3, 3))