# src/reasoning/knowledge_graph.py
import json

import networkx as nx
import matplotlib.pyplot as plt
# --- CHANGE: Import missing types ---
from typing import Dict, List, Any, Tuple
# --- END CHANGE ---
import logging
from datetime import datetime
import os
import numpy as np
import math # Import math for isnan check

from src.pipeline.pipeline import NumpyEncoder

# --- Default values moved here for clarity ---
DEFAULT_MAX_NODES = 500
DEFAULT_MAX_HISTORY = 1000

class KnowledgeGraphGenerator:
    def __init__(self, logger=None, max_nodes=DEFAULT_MAX_NODES, max_history=DEFAULT_MAX_HISTORY):
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.max_nodes = max_nodes
        self.max_history = max_history

        self.node_types = {
            'sensor': 'lightblue',
            'state': 'lightgreen',
            'rule': 'lightpink',
            'anomaly': 'salmon',
            'insight': 'lightyellow',
            'metrics': 'lightgray'
        }
        # Use a single counter for unique node IDs
        self.global_node_counter = 0

        self.sensor_history = {}
        self.sensor_correlations = {}
        self.performance_history = {}
        # Adjust layout parameter based on actual max_nodes used
        self.layout_params = {
            'k': 0.8 / math.sqrt(self.max_nodes) if self.max_nodes > 0 else 0.1, # Adjusted k heuristic
            'iterations': 100,
            'seed': 42
        }
        self.logger.info(f"KnowledgeGraphGenerator initialized with max_nodes={self.max_nodes}, max_history={self.max_history}")

    def _get_unique_node_id(self, prefix: str) -> str:
        """Generates a unique node ID."""
        self.global_node_counter += 1
        return f"{prefix}_{self.global_node_counter}"

    def _calculate_trend(self, sensor: str, value: float) -> str:
        """Calculate the trend of a sensor value based on its history."""
        if not isinstance(value, (int, float)) or math.isnan(value):
            self.logger.warning(f"Invalid value '{value}' for sensor '{sensor}' trend calculation. Skipping.")
            return " N/A"

        if sensor not in self.sensor_history:
            self.sensor_history[sensor] = []

        self.sensor_history[sensor].append(value)
        # Limit history size
        if len(self.sensor_history[sensor]) > self.max_history:
             self.sensor_history[sensor].pop(0)

        if len(self.sensor_history[sensor]) >= 2:
            last_val = self.sensor_history[sensor][-1]
            prev_val = self.sensor_history[sensor][-2]
            try:
                # Check for NaN before comparison
                if math.isnan(last_val) or math.isnan(prev_val):
                     return " N/A"
                trend_sign = np.sign(last_val - prev_val)
                if trend_sign > 0: return "↑"
                elif trend_sign < 0: return "↓"
                else: return "→"
            except TypeError:
                 self.logger.warning(f"Could not compare values for trend: {last_val}, {prev_val}")
                 return " N/A"
        else:
            return "→" # Initial state

    def _get_state_info(self, current_state_dict: Dict[str, Any]) -> Tuple[int, str]:
        """Extract state value (int) and description string."""
        state_value = 0 # Default to Normal (0)
        state_description = "Normal"
        try:
             raw_state = current_state_dict.get('system_state')
             if raw_state is not None:
                  state_value = int(float(raw_state)) # Handle potential float input
                  if state_value == 1:
                      state_description = "Degraded"
                  elif state_value == 2:
                      state_description = "Critical"
                  elif state_value == 0:
                      state_description = "Normal"
                  else:
                      state_description = f"Unknown ({state_value})"
                      state_value = 0 # Map unknown back to Normal for safety
             else:
                 self.logger.warning("system_state key missing in current_state_dict. Defaulting to Normal.")

        except (ValueError, TypeError) as e:
             self.logger.warning(f"Invalid system_state value '{current_state_dict.get('system_state')}': {e}. Defaulting to Normal.")
             state_value = 0
             state_description = "Normal (Invalid Input)"

        # Final check to ensure state_value is within expected range
        if state_value not in [0, 1, 2]:
            self.logger.warning(f"Corrected out-of-range state value {state_value} to 0.")
            state_value = 0
            state_description += " [Corrected Index]"

        return state_value, state_description


    def _calculate_correlations(self):
        """Safely calculate correlations between sensors."""
        sensors = list(self.sensor_history.keys())
        # Only calculate if we have history for at least 2 sensors
        if len(sensors) < 2:
            return

        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                s1, s2 = sensors[i], sensors[j]
                hist1 = self.sensor_history[s1]
                hist2 = self.sensor_history[s2]
                pair_key = f"{s1}_{s2}"

                # Require minimum history length for meaningful correlation
                min_hist_len = 10
                if len(hist1) >= min_hist_len and len(hist2) >= min_hist_len:
                    try:
                        # Ensure arrays are of the same length for corrcoef
                        common_len = min(len(hist1), len(hist2))
                        arr1 = np.array(hist1[-common_len:], dtype=float)
                        arr2 = np.array(hist2[-common_len:], dtype=float)

                        # Check for constant arrays (which lead to NaN correlation)
                        if np.all(arr1 == arr1[0]) or np.all(arr2 == arr2[0]):
                            self.logger.debug(f"Skipping correlation for {pair_key}: one array is constant.")
                            self.sensor_correlations.pop(pair_key, None)
                            continue

                        corr_matrix = np.corrcoef(arr1, arr2)

                        # Check if corr_matrix is valid before accessing element
                        if corr_matrix.shape == (2, 2):
                             corr = corr_matrix[0, 1]
                             if not math.isnan(corr):
                                  self.sensor_correlations[pair_key] = corr
                             else:
                                  self.logger.debug(f"Correlation between {s1} and {s2} resulted in NaN. Skipping.")
                                  self.sensor_correlations.pop(pair_key, None)
                        else:
                             self.logger.warning(f"Invalid correlation matrix shape for {pair_key}: {corr_matrix.shape}. Skipping.")
                             self.sensor_correlations.pop(pair_key, None)

                    except Exception as corr_err:
                        self.logger.warning(f"Could not calculate correlation between {s1} and {s2}: {corr_err}")
                        self.sensor_correlations.pop(pair_key, None)
                else:
                    # Not enough history, remove old value if it exists
                    self.sensor_correlations.pop(pair_key, None)


    def _add_edge_safe(self, u_node: str, v_node: str, **attrs):
        """Adds an edge only if both source and target nodes exist in the graph."""
        if u_node is not None and v_node is not None: # Also check for None IDs
            if self.graph.has_node(u_node) and self.graph.has_node(v_node):
                self.graph.add_edge(u_node, v_node, **attrs)
            else:
                self.logger.warning(f"Skipping edge ({u_node} -> {v_node}): one or both nodes not found in graph.")
        else:
             self.logger.warning(f"Skipping edge: Node ID is None (u={u_node}, v={v_node}).")


    def update_graph(self,
                     current_state: Dict[str, Any],
                     insights: List[str],
                     rules: List[Dict], # Expects list of {'rule': str, 'confidence': float}
                     anomalies: Dict[str, Any]) -> None: # Expects dict like {'severity': int, 'confidence': float}
        """Update knowledge graph with new information, adding nodes before edges."""
        # --- Prune first ---
        self._prune_old_nodes()

        try:
            timestamp_dt = datetime.now()
            timestamp_str = timestamp_dt.isoformat()

            # --- Dictionaries to hold IDs created in this step ---
            sensor_node_ids_this_step = {}
            rule_node_ids_this_step = []
            insight_node_ids_this_step = []
            state_id_this_step = None
            metrics_id_this_step = None
            anomaly_id_this_step = None
            # ---

            # Add Sensor Nodes
            sensor_keys = ['temperature', 'vibration', 'pressure'] # Define expected sensors
            for sensor in sensor_keys:
                value = current_state.get(sensor)
                if value is not None:
                    try:
                        value_f = float(value)
                        trend = self._calculate_trend(sensor, value_f)
                        node_id = self._get_unique_node_id(sensor)
                        self.graph.add_node(node_id,
                                            type='sensor',
                                            sensor_name=sensor,
                                            value=value_f,
                                            trend=trend,
                                            label=f"{sensor}\n{value_f:.2f} {trend}",
                                            color=self.node_types['sensor'],
                                            timestamp=timestamp_dt)
                        sensor_node_ids_this_step[sensor] = node_id
                    except (ValueError, TypeError) as e:
                         self.logger.warning(f"Could not process sensor '{sensor}' value '{value}': {e}")
                else:
                     self.logger.debug(f"Sensor '{sensor}' not found in current_state or value is None.")

            # Add State Node
            state_value, state_label_part = self._get_state_info(current_state)
            state_id_this_step = self._get_unique_node_id("state")
            state_labels = ['Normal', 'Degraded', 'Critical'] # Ensure this aligns with state_value indices
            self.graph.add_node(state_id_this_step,
                                type='state',
                                value=state_value,
                                label=f"State: {state_labels[state_value]}\n({state_label_part})",
                                color=self.node_types['state'],
                                timestamp=timestamp_dt)

            # Add Metrics Node
            eff_idx = current_state.get('efficiency_index')
            perf_score = current_state.get('performance_score')
            if eff_idx is not None and perf_score is not None:
                 try:
                      metrics_id_this_step = self._get_unique_node_id("metrics")
                      self.graph.add_node(metrics_id_this_step,
                                           type='metrics',
                                           efficiency=float(eff_idx),
                                           performance=float(perf_score),
                                           label=(f"Metrics\n"
                                                  f"Eff: {float(eff_idx):.2f}\n"
                                                  f"Perf: {float(perf_score):.1f}"),
                                           color=self.node_types['metrics'],
                                           timestamp=timestamp_dt)
                 except (ValueError, TypeError) as e:
                      self.logger.warning(f"Could not process metrics values (Eff:'{eff_idx}', Perf:'{perf_score}'): {e}")
                      metrics_id_this_step = None # Invalidate ID if error


            # Add Rule Nodes
            for rule_info in rules:
                if isinstance(rule_info, dict) and 'rule' in rule_info and 'confidence' in rule_info:
                    if rule_info['confidence'] > 0.7:
                        rule_id = self._get_unique_node_id("rule")
                        rule_str = rule_info['rule']
                        label_rule_str = (rule_str[:27] + '...') if len(rule_str) > 30 else rule_str
                        self.graph.add_node(rule_id,
                                            type='rule',
                                            rule_string=rule_str,
                                            confidence=float(rule_info['confidence']),
                                            label=f"Rule\n{label_rule_str}\nConf: {rule_info['confidence']:.2f}",
                                            color=self.node_types['rule'],
                                            timestamp=timestamp_dt)
                        rule_node_ids_this_step.append(rule_id)
                else:
                    self.logger.warning(f"Skipping invalid rule format during node creation: {rule_info}")

            # Add Anomaly Node
            anomaly_severity = int(anomalies.get('severity', 0))
            if anomaly_severity > 0:
                anomaly_id_this_step = self._get_unique_node_id("anomaly")
                self.graph.add_node(anomaly_id_this_step,
                                    type='anomaly',
                                    severity=anomaly_severity,
                                    confidence=float(anomalies.get('confidence', 0.0)),
                                    label=f"Anomaly\nSeverity: {anomaly_severity}",
                                    color=self.node_types['anomaly'],
                                    timestamp=timestamp_dt)

            # Add Insight Nodes
            for insight_str in insights:
                insight_id = self._get_unique_node_id("insight")
                label_insight_str = (insight_str[:47] + '...') if len(insight_str) > 50 else insight_str
                self.graph.add_node(insight_id,
                                    type='insight',
                                    text=insight_str,
                                    label=f"Insight\n{label_insight_str}",
                                    color=self.node_types['insight'],
                                    timestamp=timestamp_dt)
                insight_node_ids_this_step.append(insight_id)

            # --- Add Edges using _add_edge_safe and IDs from this step ---
            edge_ts = timestamp_str # Timestamp for edges

            # Sensor -> State
            for sensor_id in sensor_node_ids_this_step.values():
                 self._add_edge_safe(sensor_id, state_id_this_step, weight=0.8, label='influences', timestamp=edge_ts)

            # State -> Metrics
            self._add_edge_safe(state_id_this_step, metrics_id_this_step, weight=1.0, label='has', timestamp=edge_ts)

            # Rule Connections
            for rule_id in rule_node_ids_this_step:
                 rule_data = self.graph.nodes[rule_id] # Get data added above
                 rule_string_lower = rule_data.get('rule_string', '').lower()
                 rule_confidence = rule_data.get('confidence', 0.0)

                 # Sensor -> Rule
                 for sensor_name, sensor_id in sensor_node_ids_this_step.items():
                     # More specific check for sensor involvement in rule body
                     if f"{sensor_name}(" in rule_string_lower or f" {sensor_name}_" in rule_string_lower:
                         self._add_edge_safe(sensor_id, rule_id, weight=rule_confidence, label='related_to', timestamp=edge_ts)

                 # Rule -> State (Basic implication check)
                 rule_head = rule_string_lower.split(":-")[0].strip()
                 if "degraded_state" in rule_head:
                     self._add_edge_safe(rule_id, state_id_this_step, weight=rule_confidence, label='implies_degraded', timestamp=edge_ts)
                 elif "critical_state" in rule_head:
                     self._add_edge_safe(rule_id, state_id_this_step, weight=rule_confidence, label='implies_critical', timestamp=edge_ts)
                 # Add more rule->state links based on rule heads if needed

            # Anomaly Connections
            if anomaly_id_this_step:
                 # State -> Anomaly
                 self._add_edge_safe(state_id_this_step, anomaly_id_this_step, weight=1.0, label='indicates', timestamp=edge_ts)
                 # Rule -> Anomaly
                 for rule_id in rule_node_ids_this_step:
                     rule_conf = self.graph.nodes[rule_id].get('confidence', 0.0)
                     self._add_edge_safe(rule_id, anomaly_id_this_step, weight=rule_conf, label='detects', timestamp=edge_ts)

            # Insight Connections
            for insight_id in insight_node_ids_this_step:
                 # State -> Insight
                 self._add_edge_safe(state_id_this_step, insight_id, weight=0.5, label='context_for', timestamp=edge_ts)
                 # Anomaly -> Insight
                 self._add_edge_safe(anomaly_id_this_step, insight_id, weight=1.0, label='explained_by', timestamp=edge_ts) # Safe even if anomaly_id is None
                 # Rule -> Insight (Connect relevant rules)
                 insight_text_lower = self.graph.nodes[insight_id].get('text', '').lower()
                 for rule_id in rule_node_ids_this_step:
                      rule_head = self.graph.nodes[rule_id].get('rule_string', '').split(":-")[0].strip().lower()
                      # Check if rule head (name) appears in the insight text
                      if rule_head and rule_head in insight_text_lower:
                           self._add_edge_safe(rule_id, insight_id, weight=0.7, label='generates', timestamp=edge_ts)

            # Correlation Edges (between latest sensor nodes)
            self._calculate_correlations() # Recalculate based on updated history
            for pair, corr in self.sensor_correlations.items():
                 if abs(corr) > 0.7:
                     s1_name, s2_name = pair.split('_')
                     # Use the IDs generated in *this* step
                     s1_id = sensor_node_ids_this_step.get(s1_name)
                     s2_id = sensor_node_ids_this_step.get(s2_name)
                     self._add_edge_safe(s1_id, s2_id, weight=abs(corr), label=f'corr: {corr:.2f}', style='dotted', timestamp=edge_ts)


            self.logger.info(f"Knowledge graph updated: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {e}", exc_info=True)
            # Don't raise, allow continuation


    def _prune_old_nodes(self) -> None:
        """Remove oldest nodes if graph exceeds size limit, preserving structure if possible."""
        try:
            num_nodes = len(self.graph)
            if num_nodes > self.max_nodes:
                num_to_remove = num_nodes - self.max_nodes
                # Get nodes with timestamp attribute, ensure it's datetime
                nodes_with_ts = [
                    (n, data['timestamp']) for n, data in self.graph.nodes(data=True)
                    if 'timestamp' in data and isinstance(data['timestamp'], datetime)
                ]

                if not nodes_with_ts or len(nodes_with_ts) <= num_to_remove:
                    self.logger.warning(f"Not enough nodes with valid timestamps ({len(nodes_with_ts)}) to prune {num_to_remove} nodes. Skipping prune.")
                    return

                # Sort by timestamp (oldest first)
                nodes_with_ts.sort(key=lambda x: x[1])

                # Identify nodes to remove
                nodes_to_remove_ids = [n[0] for n in nodes_with_ts[:num_to_remove]]

                # Remove the identified nodes
                self.graph.remove_nodes_from(nodes_to_remove_ids)
                self.logger.info(f"Pruned {len(nodes_to_remove_ids)} old nodes from graph (Limit: {self.max_nodes}).")

        except Exception as e:
            self.logger.error(f"Error pruning nodes: {e}", exc_info=True)


    def visualize(self, output_path: str) -> None:
        """Generate and save knowledge graph visualization."""
        if not self.graph or self.graph.number_of_nodes() == 0:
             self.logger.warning("Graph is empty, skipping visualization.")
             return

        try:
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            self.logger.info(f"Starting visualization with {num_nodes} nodes and {num_edges} edges.")
            node_types_present = set(nx.get_node_attributes(self.graph, 'type').values())
            self.logger.info(f"Node types present: {node_types_present}")

            # Adjust figure size based on node count
            figsize_base = 15
            figsize_scale = max(1.0, num_nodes / 50.0) # Scale up moderately
            plt.figure(figsize=(figsize_base * figsize_scale, figsize_base * figsize_scale * 0.75))

            # Layout calculation
            try:
                 # Adaptive k based on node count
                 effective_k = 1.0 / math.sqrt(num_nodes) if num_nodes > 0 else 0.1
                 pos = nx.spring_layout(self.graph, k=effective_k, iterations=self.layout_params['iterations'], seed=self.layout_params['seed'])
            except Exception as layout_error:
                 self.logger.warning(f"Spring layout failed: {layout_error}. Falling back to random layout.")
                 pos = nx.random_layout(self.graph, seed=self.layout_params['seed'])

            # Node sizes and colors
            sizes = {'sensor': 2500, 'state': 4000, 'rule': 2000, 'insight': 2200, 'anomaly': 3500, 'metrics': 2500}
            default_size = 1500
            node_size_list = [sizes.get(self.graph.nodes[n].get('type', ''), default_size) for n in self.graph.nodes()]
            node_color_list = [self.node_types.get(self.graph.nodes[n].get('type', ''), 'gray') for n in self.graph.nodes()]

            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_color_list, node_size=node_size_list, alpha=0.8)

            # Draw edges
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
                'corr': {'style': 'dotted', 'width': 1.5, 'color': 'blue'} # Style for correlation
            }
            default_edge_style = {'style': 'solid', 'width': 0.5, 'color': 'lightgray'}

            for u, v, data in self.graph.edges(data=True):
                 edge_label = data.get('label', '')
                 # Special handling for correlation label format
                 base_label = 'corr' if edge_label.startswith('corr:') else edge_label
                 style_attrs = edge_styles.get(base_label, default_edge_style)

                 # Scale width by edge weight if present, default to style width otherwise
                 edge_weight = data.get('weight', 1.0)
                 draw_width = style_attrs['width'] * (edge_weight if edge_weight is not None else 1.0)

                 nx.draw_networkx_edges(self.graph, pos,
                                        edgelist=[(u, v)],
                                        style=style_attrs['style'],
                                        width=max(0.5, draw_width), # Ensure minimum width
                                        edge_color=style_attrs['color'],
                                        arrows=True,
                                        arrowsize=15,
                                        alpha=0.6,
                                        connectionstyle='arc3,rad=0.1')

            # Draw labels
            labels = {n: data.get('label', n) for n, data in self.graph.nodes(data=True)}
            font_size = max(4, 8 - int(num_nodes / 50)) # Decrease font size for large graphs
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=font_size, font_weight='normal')

            # Draw edge labels (optional)
            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            if num_edges < 150: # Only draw edge labels for smaller graphs to avoid clutter
                nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=max(3, font_size-2), font_color='dimgray')

            plt.title(f"ANSR-DT Knowledge Graph ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", size=16)
            plt.axis('off')
            # Use tight_layout cautiously, might fail on very large graphs
            try:
                 plt.tight_layout()
            except ValueError:
                 self.logger.warning("tight_layout failed, proceeding without it.")

            # Save figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Knowledge graph visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}", exc_info=True)
            plt.close() # Ensure plot is closed even on error


    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current graph state."""
        try:
            num_nodes = self.graph.number_of_nodes()
            if num_nodes == 0:
                return {'total_nodes': 0, 'total_edges': 0, 'node_types': {}, 'avg_degree': 0.0, 'timestamp': datetime.now().isoformat()}

            node_types_counts = {
                node_type: len([n for n, d in self.graph.nodes(data=True)
                                if d.get('type') == node_type])
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
            # Use networkx's built-in JSON exporter
            graph_data = nx.node_link_data(self.graph)

            # Add metadata
            graph_data['metadata'] = {
                 'timestamp': datetime.now().isoformat(),
                 'statistics': self.get_graph_statistics()
            }

            # Ensure all datetime objects are converted to strings
            for node in graph_data.get('nodes', []):
                 for key, value in node.items():
                      if isinstance(value, datetime):
                           node[key] = value.isoformat()
            for link in graph_data.get('links', []):
                 for key, value in link.items():
                      if isinstance(value, datetime):
                           link[key] = value.isoformat()

            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2, cls=NumpyEncoder) # Use encoder for safety

            self.logger.info(f"Graph data exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}", exc_info=True)
            # Avoid raising here unless critical