# src/reasoning/knowledge_graph.py
import json
import networkx as nx
import matplotlib.pyplot as plt
import re  # Added for improved sensor name matching
from typing import Dict, List, Any, Tuple, Optional  # Ensure Optional and Tuple are imported
import logging
from datetime import datetime
import os
import numpy as np
import math  # Import math for isnan check

# Attempt to import NumpyEncoder, define locally if import fails
try:
    # Assuming NumpyEncoder is defined in pipeline.py and handles np types + datetime
    from src.pipeline.pipeline import NumpyEncoder
except ImportError:
    logging.warning("Could not import NumpyEncoder from pipeline, defining locally.")
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON Encoder that converts NumPy data types and datetime to native Python types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                 # Handle NaN and Inf for JSON compatibility
                 if np.isnan(obj):
                     return None
                 if np.isinf(obj):
                     return None  # Or use a large number string like 'Infinity'
                 return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle datetime objects
            elif isinstance(obj, datetime):
                 return obj.isoformat()
            return super(NumpyEncoder, self).default(obj)

# Default values for graph parameters
DEFAULT_MAX_NODES = 500  # Increased default limit
DEFAULT_MAX_HISTORY = 1000

class KnowledgeGraphGenerator:
    """Generates and visualizes a knowledge graph representing system state, insights, and rules."""

    def __init__(self, logger: Optional[logging.Logger] = None, max_nodes: int = DEFAULT_MAX_NODES, max_history: int = DEFAULT_MAX_HISTORY):
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.max_nodes = max_nodes
        self.max_history = max_history

        # Parameterize correlation settings
        self.correlation_threshold = 0.7
        self.min_history_for_correlation = 10  # Min points needed for correlation calc

        # Node styling
        self.node_types = {
            'sensor': 'lightblue',
            'state': 'lightgreen',
            'rule': 'lightpink',
            'anomaly': 'salmon',
            'insight': 'lightyellow',
            'metrics': 'lightgray'
        }
        # Shared counter for generating unique node IDs
        self.global_node_counter = 0

        # Data storage for trends and correlations
        self.sensor_history: Dict[str, List[float]] = {}
        self.sensor_correlations: Dict[str, float] = {}

        # Layout parameters for visualization
        # Adjust layout parameter 'k' based on max_nodes for better initial spacing
        self.layout_params = {
            'k': 0.8 / math.sqrt(self.max_nodes) if self.max_nodes > 0 else 0.1,
            'iterations': 100,  # Keep iterations reasonable
            'seed': 42  # For reproducible layouts
        }
        # Initialize caching for layout positions to speed up repeated visualizations
        self._cached_pos: Optional[Dict[Any, Any]] = None
        self._graph_hash: Optional[str] = None  # Use robust graph hash for cache invalidation

        self.logger.info(f"KnowledgeGraphGenerator initialized with max_nodes={self.max_nodes}, max_history={self.max_history}")

    def _get_unique_node_id(self, prefix: str) -> str:
        """Generates a unique node ID based on a prefix and a global counter."""
        self.global_node_counter += 1
        return f"{prefix}_{self.global_node_counter}"

    def _calculate_trend(self, sensor: str, value: float) -> str:
        """Calculates the trend (↑, ↓, →, N/A) of a sensor value based on history."""
        # Validate input value
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            self.logger.debug(f"Invalid value '{value}' for sensor '{sensor}' trend calculation.")
            return "N/A"

        if sensor not in self.sensor_history:
            self.sensor_history[sensor] = []

        self.sensor_history[sensor].append(value)
        # Maintain history window size
        if len(self.sensor_history[sensor]) > self.max_history:
             self.sensor_history[sensor] = self.sensor_history[sensor][-self.max_history:]  # Efficient slice

        # Determine trend based on the last two points
        if len(self.sensor_history[sensor]) >= 2:
            last_val = self.sensor_history[sensor][-1]
            prev_val = self.sensor_history[sensor][-2]
            try:
                # Check for NaN/Inf before comparison
                if math.isnan(last_val) or math.isnan(prev_val) or math.isinf(last_val) or math.isinf(prev_val):
                    return "N/A"
                diff = last_val - prev_val
                tolerance = 1e-6  # Define a small tolerance for float comparison
                if diff > tolerance:
                    return "↑"
                elif diff < -tolerance:
                    return "↓"
                else:
                    return "→"
            except TypeError:
                 self.logger.warning(f"Could not compare values for trend calculation: {last_val}, {prev_val}")
                 return "N/A"
        else:
            return "→"  # Default trend for single point or start

    def _get_state_info(self, current_state_dict: Dict[str, Any]) -> Tuple[int, str]:
        """Safely extracts system state value (int) and description string."""
        state_value = 0  # Default to Normal (0)
        state_description = "Normal"
        raw_state = current_state_dict.get('system_state')

        if raw_state is not None:
            try:
                 state_value = int(float(raw_state))  # Convert safely, allowing float input
                 if state_value == 1:
                     state_description = "Degraded"
                 elif state_value == 2:
                     state_description = "Critical"
                 elif state_value == 0:
                     state_description = "Normal"
                 else:  # Handle unexpected state values
                     state_description = f"Unknown ({state_value})"
                     self.logger.warning(f"Received unknown system_state value: {state_value}. Treating as Normal.")
                     state_value = 0  # Map unknown back to Normal for safety
            except (ValueError, TypeError) as e:
                 self.logger.warning(f"Invalid system_state value '{raw_state}': {e}. Defaulting to Normal.")
                 state_value = 0
                 state_description = "Normal (Invalid Input)"
        else:
            self.logger.warning("'system_state' key missing in current_state_dict provided to KG. Defaulting to Normal.")
            state_value = 0
            state_description = "Normal (Missing Key)"

        # Final check to ensure state_value is a valid index for labels list later
        if state_value not in [0, 1, 2]:
            self.logger.error(f"Internal Error: State value {state_value} is out of bounds [0, 1, 2] after processing. Forcing to 0.")
            state_value = 0  # Force to valid index
            state_description += " [Corrected Index]"

        return state_value, state_description

    def _calculate_correlations(self):
        """Safely calculate correlations between sensors with sufficient history."""
        sensors = list(self.sensor_history.keys())
        if len(sensors) < 2:
            return

        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                s1, s2 = sensors[i], sensors[j]
                hist1 = self.sensor_history.get(s1, [])
                hist2 = self.sensor_history.get(s2, [])
                pair_key = f"{s1}_{s2}"

                common_len = min(len(hist1), len(hist2))
                if common_len >= self.min_history_for_correlation:
                    try:
                        arr1 = np.array(hist1[-common_len:], dtype=float)
                        arr2 = np.array(hist2[-common_len:], dtype=float)

                        # Pre-check for NaN/Inf and constant arrays
                        if np.isnan(arr1).any() or np.isnan(arr2).any() or \
                           np.isinf(arr1).any() or np.isinf(arr2).any():
                            self.logger.debug(f"Skipping correlation for {pair_key}: NaN/Inf values found.")
                            self.sensor_correlations.pop(pair_key, None)
                            continue
                        if np.std(arr1) < 1e-9 or np.std(arr2) < 1e-9:
                            self.logger.debug(f"Skipping correlation for {pair_key}: near-zero variance.")
                            self.sensor_correlations.pop(pair_key, None)
                            continue

                        corr_matrix = np.corrcoef(arr1, arr2)

                        if corr_matrix.shape == (2, 2):
                             corr = corr_matrix[0, 1]
                             if not math.isnan(corr):  # Final check on result
                                  self.sensor_correlations[pair_key] = corr
                             else:
                                  self.logger.debug(f"Correlation calc for {pair_key} resulted in NaN despite checks. Removing.")
                                  self.sensor_correlations.pop(pair_key, None)
                        else:
                             self.logger.warning(f"Invalid corr matrix shape {corr_matrix.shape} for {pair_key}.")
                             self.sensor_correlations.pop(pair_key, None)
                    except Exception as corr_err:
                        self.logger.warning(f"Error calculating correlation between {s1} and {s2}: {corr_err}", exc_info=True)
                        self.sensor_correlations.pop(pair_key, None)  # Remove potentially invalid entry
                else:
                    # Not enough history, ensure no stale correlation exists
                    self.sensor_correlations.pop(pair_key, None)

    def _add_edge_safe(self, u_node: Optional[str], v_node: Optional[str], **attrs):
        """Adds an edge only if both source and target nodes exist and are not None."""
        if u_node is None or v_node is None:
             self.logger.debug(f"Skipping edge: Source or target node ID is None (u={u_node}, v={v_node}).")
             return

        if self.graph.has_node(u_node) and self.graph.has_node(v_node):
            self.graph.add_edge(u_node, v_node, **attrs)
        else:
            missing_nodes = []
            if not self.graph.has_node(u_node):
                missing_nodes.append(f"source '{u_node}'")
            if not self.graph.has_node(v_node):
                missing_nodes.append(f"target '{v_node}'")
            self.logger.warning(f"Skipping edge ({u_node} -> {v_node}): {', '.join(missing_nodes)} not found in graph.")

    def update_graph(self,
                     current_state: Dict[str, Any],
                     insights: List[str],
                     rules: List[Dict],
                     anomalies: Dict[str, Any]) -> None:
        """Update knowledge graph as a snapshot of current state.

        This update clears previous nodes/edges so that each update reflects only the current inference.
        """
        # Clear the graph and reset the node counter to provide a fresh snapshot.
        self.graph.clear()
        self.global_node_counter = 0

        # (Optionally, _prune_old_nodes() is not needed when clearing the graph.)
        # Ensure required keys are present in current_state by assigning defaults if missing.
        required_keys = ['temperature', 'vibration', 'pressure', 'system_state', 'efficiency_index', 'performance_score']
        defaults = {
            'temperature': 70.0,
            'vibration': 50.0,
            'pressure': 30.0,
            'system_state': 0,
            'efficiency_index': 0.9,
            'performance_score': 90.0
        }
        for key in required_keys:
            if key not in current_state or current_state[key] is None:
                self.logger.warning(f"Current state missing key '{key}', assigning default value {defaults[key]}.")
                current_state[key] = defaults[key]

        try:
            timestamp_dt = datetime.now()
            timestamp_str = timestamp_dt.isoformat()  # For edge attributes if needed

            # --- Store IDs created in THIS update step ---
            sensor_node_ids: Dict[str, str] = {}
            rule_node_ids: List[str] = []
            insight_node_ids: List[str] = []
            state_id: Optional[str] = None
            metrics_id: Optional[str] = None
            anomaly_id: Optional[str] = None

            # --- 1. Add Nodes (with validity checks) ---

            # Sensor Nodes
            sensor_keys = ['temperature', 'vibration', 'pressure']
            for sensor in sensor_keys:
                value = current_state.get(sensor)
                if value is not None:
                    try:
                        value_f = float(value)
                        if math.isnan(value_f) or math.isinf(value_f):
                            raise ValueError("Invalid float value")
                        trend = self._calculate_trend(sensor, value_f)
                        node_id = self._get_unique_node_id(sensor)
                        self.graph.add_node(node_id, type='sensor', sensor_name=sensor, value=value_f, trend=trend,
                                            label=f"{sensor}\n{value_f:.2f} {trend}", color=self.node_types['sensor'], timestamp=timestamp_dt)
                        sensor_node_ids[sensor] = node_id
                    except (ValueError, TypeError) as e:
                         self.logger.warning(f"Could not process sensor '{sensor}' value '{value}': {e}. Skipping node.")
                else:
                     self.logger.debug(f"Sensor '{sensor}' missing or None in current_state.")

            # State Node (Always created, uses safe extraction)
            state_value, state_label_part = self._get_state_info(current_state)
            state_id = self._get_unique_node_id("state")
            state_labels = ['Normal', 'Degraded', 'Critical']
            self.graph.add_node(state_id, type='state', value=state_value,
                                label=f"State: {state_labels[state_value]}\n({state_label_part})",
                                color=self.node_types['state'], timestamp=timestamp_dt)

            # Metrics Node (Only if data is valid)
            eff_idx = current_state.get('efficiency_index')
            perf_score = current_state.get('performance_score')
            if eff_idx is not None and perf_score is not None:
                 try:
                      eff_f, perf_f = float(eff_idx), float(perf_score)
                      if math.isnan(eff_f) or math.isinf(eff_f) or math.isnan(perf_f) or math.isinf(perf_f):
                          raise ValueError("Invalid float value for metrics")
                      metrics_id = self._get_unique_node_id("metrics")
                      self.graph.add_node(metrics_id, type='metrics', efficiency=eff_f, performance=perf_f,
                                           label=f"Metrics\nEff: {eff_f:.2f}\nPerf: {perf_f:.1f}",
                                           color=self.node_types['metrics'], timestamp=timestamp_dt)
                 except (ValueError, TypeError) as e:
                      self.logger.warning(f"Could not process metrics values (Eff:'{eff_idx}', Perf:'{perf_score}'): {e}")
            # Rule Nodes
            for rule_info in rules:
                if isinstance(rule_info, dict) and 'rule' in rule_info and 'confidence' in rule_info:
                    try:
                        confidence = float(rule_info['confidence'])
                        if confidence > 0.7:  # Only add high-confidence rules to graph
                            rule_id = self._get_unique_node_id("rule")
                            rule_str = rule_info['rule']
                            label_rule_str = (rule_str[:27] + '...') if len(rule_str) > 30 else rule_str
                            self.graph.add_node(rule_id, type='rule', rule_string=rule_str, confidence=confidence,
                                                label=f"Rule\n{label_rule_str}\nConf: {confidence:.2f}",
                                                color=self.node_types['rule'], timestamp=timestamp_dt)
                            rule_node_ids.append(rule_id)
                    except (ValueError, TypeError):
                         self.logger.warning(f"Invalid confidence '{rule_info.get('confidence')}' for rule. Skipping rule node.")
                else:
                    self.logger.warning(f"Skipping invalid rule format during node creation: {rule_info}")

            # Anomaly Node
            anomaly_severity = int(anomalies.get('severity', 0))
            if anomaly_severity > 0:
                anomaly_id = self._get_unique_node_id("anomaly")
                self.graph.add_node(anomaly_id, type='anomaly', severity=anomaly_severity,
                                    confidence=float(anomalies.get('confidence', 0.0)),
                                    label=f"Anomaly\nSeverity: {anomaly_severity}",
                                    color=self.node_types['anomaly'], timestamp=timestamp_dt)

            # Insight Nodes
            for insight_str in insights:
                 if isinstance(insight_str, str):
                    insight_id = self._get_unique_node_id("insight")
                    label_insight_str = (insight_str[:47] + '...') if len(insight_str) > 50 else insight_str
                    self.graph.add_node(insight_id, type='insight', text=insight_str,
                                        label=f"Insight\n{label_insight_str}",
                                        color=self.node_types['insight'], timestamp=timestamp_dt)
                    insight_node_ids.append(insight_id)

            # --- 2. Add Edges (using generated IDs and _add_edge_safe) ---
            edge_ts = timestamp_str

            # Sensor -> State
            for sensor_id in sensor_node_ids.values():
                 self._add_edge_safe(sensor_id, state_id, weight=0.8, label='influences', timestamp=edge_ts)

            # State -> Metrics
            self._add_edge_safe(state_id, metrics_id, weight=1.0, label='has', timestamp=edge_ts)

            # Rule Connections
            for rule_id in rule_node_ids:
                 rule_data = self.graph.nodes[rule_id]
                 rule_string_lower = rule_data.get('rule_string', '').lower()
                 rule_confidence = rule_data.get('confidence', 0.0)
                 # Sensor -> Rule (Improved check using regex)
                 for sensor_name, sensor_id in sensor_node_ids.items():
                     pattern = r'\b' + re.escape(sensor_name) + r'\b'
                     pattern_pred = r'\b' + re.escape(sensor_name) + r'\s*\('
                     if re.search(pattern, rule_string_lower) or re.search(pattern_pred, rule_string_lower):
                         self._add_edge_safe(sensor_id, rule_id, weight=rule_confidence, label='related_to', timestamp=edge_ts)
                 # Rule -> State (Based on rule head content)
                 rule_head = rule_string_lower.split(":-")[0].strip()
                 if "degraded" in rule_head:
                      self._add_edge_safe(rule_id, state_id, weight=rule_confidence, label='implies_degraded', timestamp=edge_ts)
                 elif "critical" in rule_head:
                      self._add_edge_safe(rule_id, state_id, weight=rule_confidence, label='implies_critical', timestamp=edge_ts)

            # Anomaly Connections
            if anomaly_id:
                 self._add_edge_safe(state_id, anomaly_id, weight=1.0, label='indicates', timestamp=edge_ts)
                 for rule_id in rule_node_ids:
                     rule_conf = self.graph.nodes[rule_id].get('confidence', 0.0)
                     self._add_edge_safe(rule_id, anomaly_id, weight=rule_conf, label='detects', timestamp=edge_ts)

            # Insight Connections
            for insight_id in insight_node_ids:
                 self._add_edge_safe(state_id, insight_id, weight=0.5, label='context_for', timestamp=edge_ts)
                 self._add_edge_safe(anomaly_id, insight_id, weight=1.0, label='explained_by', timestamp=edge_ts)
                 insight_text_lower = self.graph.nodes[insight_id].get('text', '').lower()
                 for rule_id in rule_node_ids:
                      rule_head = self.graph.nodes[rule_id].get('rule_string', '').split(":-")[0].strip().lower()
                      if rule_head and rule_head in insight_text_lower:
                           self._add_edge_safe(rule_id, insight_id, weight=0.7, label='generates', timestamp=edge_ts)

            # Correlation Edges
            self._calculate_correlations()
            for pair, corr in self.sensor_correlations.items():
                 if abs(corr) > self.correlation_threshold:
                     s1_name, s2_name = pair.split('_')
                     s1_id = sensor_node_ids.get(s1_name)
                     s2_id = sensor_node_ids.get(s2_name)
                     self._add_edge_safe(s1_id, s2_id, weight=abs(corr), label=f'corr: {corr:.2f}', style='dotted', timestamp=edge_ts)

            self.logger.info(f"Knowledge graph updated: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        except Exception as e:
            self.logger.error(f"Critical error updating knowledge graph: {e}", exc_info=True)

    def _prune_old_nodes(self) -> None:
        """Removes oldest nodes if graph exceeds size limit."""
        try:
            num_nodes = self.graph.number_of_nodes()
            if num_nodes > self.max_nodes:
                num_to_remove = num_nodes - self.max_nodes
                nodes_with_ts = [
                    (n, data['timestamp']) for n, data in self.graph.nodes(data=True)
                    if isinstance(data.get('timestamp'), datetime)
                ]

                if len(nodes_with_ts) <= num_to_remove:
                    self.logger.warning(f"Cannot prune: Not enough nodes ({len(nodes_with_ts)}) with valid timestamps to remove {num_to_remove}.")
                    return

                nodes_with_ts.sort(key=lambda x: x[1])
                nodes_to_remove_ids = [n[0] for n in nodes_with_ts[:num_to_remove]]

                self.graph.remove_nodes_from(nodes_to_remove_ids)
                self.logger.info(f"Pruned {len(nodes_to_remove_ids)} old nodes from graph (Limit: {self.max_nodes}). New size: {self.graph.number_of_nodes()}")
        except Exception as e:
            self.logger.error(f"Error during graph pruning: {e}", exc_info=True)

    def visualize(self, output_path: str) -> None:
        """Generates and saves knowledge graph visualization."""
        if not self.graph or self.graph.number_of_nodes() == 0:
             self.logger.warning("Graph is empty, skipping visualization.")
             return

        try:
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            self.logger.info(f"Starting visualization with {num_nodes} nodes and {num_edges} edges.")
            node_types_present = set(nx.get_node_attributes(self.graph, 'type').values())
            self.logger.info(f"Node types present: {node_types_present}")

            # Adjust figure size and layout parameter 'k' based on node count
            figsize_base = 15
            figsize_scale = min(max(1.0, num_nodes / 50.0), 3.0)  # Cap scaling factor
            plt.figure(figsize=(figsize_base * figsize_scale, figsize_base * figsize_scale * 0.75))

            # Layout calculation with caching using Weisfeiler-Lehman graph hash
            current_graph_hash = nx.weisfeiler_lehman_graph_hash(self.graph)
            if self._cached_pos is not None and self._graph_hash == current_graph_hash:
                pos = self._cached_pos
                self.logger.debug("Using cached layout positions.")
            else:
                self.logger.debug("Calculating new layout positions.")
                effective_k = 1.0 / math.sqrt(num_nodes) if num_nodes > 1 else 0.5
                try:
                    pos = nx.spring_layout(self.graph, k=effective_k, iterations=self.layout_params['iterations'], seed=self.layout_params['seed'])
                except Exception as layout_error:
                    self.logger.warning(f"Spring layout failed: {layout_error}. Falling back to random layout.")
                    pos = nx.random_layout(self.graph, seed=self.layout_params['seed'])
                self._cached_pos = pos
                self._graph_hash = current_graph_hash

            sizes = {'sensor': 2500, 'state': 4000, 'rule': 2000, 'insight': 2200, 'anomaly': 3500, 'metrics': 2500}
            default_size = 1500
            node_size_list = [sizes.get(data.get('type', ''), default_size) for node, data in self.graph.nodes(data=True)]
            node_color_list = [self.node_types.get(data.get('type', ''), 'gray') for node, data in self.graph.nodes(data=True)]

            nx.draw_networkx_nodes(self.graph, pos, node_color=node_color_list, node_size=node_size_list, alpha=0.8)

            edge_styles = {
                'influences': {'style': 'solid', 'width': 1.5, 'color': 'darkgray'},
                'related_to': {'style': 'dashed', 'width': 0.8, 'color': 'gray'},
                'implies_degraded': {'style': 'solid', 'width': 1.5, 'color': 'orange'},
                'implies_critical': {'style': 'solid', 'width': 2.0, 'color': 'red'},
                'indicates': {'style': 'solid', 'width': 1.5, 'color': 'salmon'},
                'detects': {'style': 'dashed', 'width': 1.0, 'color': 'salmon'},
                'context_for': {'style': 'dotted', 'width': 0.8, 'color': 'olive'},
                'explained_by': {'style': 'solid', 'width': 1.0, 'color': 'olive'},
                'generates': {'style': 'dotted', 'width': 0.8, 'color': 'purple'},
                'has': {'style': 'solid', 'width': 1.0, 'color': 'black'},
                'corr': {'style': 'dotted', 'width': 1.5, 'color': 'blue'}
            }
            default_edge_style = {'style': 'solid', 'width': 0.5, 'color': 'lightgray'}

            for u, v, data in self.graph.edges(data=True):
                 edge_label = data.get('label', '')
                 base_label = 'corr' if edge_label.startswith('corr:') else edge_label
                 style_attrs = edge_styles.get(base_label, default_edge_style)
                 edge_weight = data.get('weight', 1.0)
                 try:
                      weight_multiplier = float(edge_weight) if edge_weight is not None else 1.0
                 except (ValueError, TypeError):
                      weight_multiplier = 1.0
                 draw_width = style_attrs['width'] * weight_multiplier

                 nx.draw_networkx_edges(self.graph, pos, edgelist=[(u, v)], style=style_attrs['style'],
                                        width=max(0.5, draw_width), edge_color=style_attrs['color'],
                                        arrows=True, arrowsize=15, alpha=0.6, connectionstyle='arc3,rad=0.1')

            labels = {n: data.get('label', n) for n, data in self.graph.nodes(data=True)}
            font_size = max(4, 8 - int(num_nodes / 50))
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=font_size, font_weight='normal')

            if num_edges < 150:
                 edge_labels = nx.get_edge_attributes(self.graph, 'label')
                 nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=max(3, font_size-2), font_color='dimgray')

            plt.title(f"ANSR-DT Knowledge Graph ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", size=16)
            plt.axis('off')
            try:
                plt.tight_layout()
            except ValueError:
                self.logger.warning("tight_layout failed.")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Knowledge graph visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}", exc_info=True)
            plt.close()  # Ensure plot is closed

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current graph state."""
        try:
            num_nodes = self.graph.number_of_nodes()
            if num_nodes == 0:
                return {'total_nodes': 0, 'total_edges': 0, 'node_types': {}, 'avg_degree': 0.0, 'timestamp': datetime.now().isoformat()}

            node_types_counts = {
                node_type: len([n for n, d in self.graph.nodes(data=True) if d.get('type') == node_type])
                for node_type in self.node_types
            }
            degrees = [d for n, d in self.graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0.0

            stats = {
                'total_nodes': num_nodes,
                'total_edges': self.graph.number_of_edges(),
                'node_types': node_types_counts,
                'avg_degree': float(avg_degree),
                'timestamp': datetime.now().isoformat()
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating graph statistics: {e}", exc_info=True)
            return {}

    def export_graph(self, output_path: str) -> None:
        """Export graph data to JSON format using networkx node_link_data."""
        try:
            graph_data = nx.node_link_data(self.graph)

            graph_data['metadata'] = {
                 'timestamp': datetime.now().isoformat(),
                 'statistics': self.get_graph_statistics()
            }

            for node_dict in graph_data.get('nodes', []):
                 for key, value in node_dict.items():
                      if isinstance(value, datetime):
                          node_dict[key] = value.isoformat()
                      elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                          node_dict[key] = int(value)
                      elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                          node_dict[key] = float(value)
                      elif isinstance(value, np.ndarray):
                          node_dict[key] = value.tolist()

            for link_dict in graph_data.get('links', []):
                 for key, value in link_dict.items():
                      if isinstance(value, datetime):
                          link_dict[key] = value.isoformat()
                      elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                          link_dict[key] = int(value)
                      elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                          link_dict[key] = float(value)
                      elif isinstance(value, np.ndarray):
                          link_dict[key] = value.tolist()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2, cls=NumpyEncoder)

            self.logger.info(f"Graph data exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}", exc_info=True)
