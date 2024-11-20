# src/reasoning/knowledge_graph.py

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
from datetime import datetime
import os
import numpy as np


class KnowledgeGraphGenerator:
    def __init__(self, logger=None, max_history=1000):
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.max_history = max_history
        # Update color definitions - remove # prefix
        self.node_types = {
            'sensor': 'lightblue',
            'state': 'lightgreen',
            'rule': 'lightpink',
            'anomaly': 'salmon',
            'insight': 'lightyellow',
            'metrics': 'lightgray'  # Add metrics node type
        }
        self.node_counter = {
            'sensor': 0,
            'state': 0,
            'rule': 0,
            'anomaly': 0,
            'insight': 0
        }
        self.sensor_history = {}  # Store historical sensor data for trend calculation
        self.sensor_correlations = {}  # Store sensor correlations
        self.performance_history = {}  # Store performance metrics history
        self.layout_params = {
            'k': 2,
            'iterations': 100,
            'seed': 42  # For reproducibility
        }

    def _calculate_trend(self, sensor: str, value: float) -> str:
        """Calculate the trend of a sensor value based on its history."""
        if sensor not in self.sensor_history:
            self.sensor_history[sensor] = []
        self.sensor_history[sensor].append(value)
        if len(self.sensor_history[sensor]) >= 2:
            trend = np.sign(self.sensor_history[sensor][-1] - self.sensor_history[sensor][-2])
            if trend == 1:
                return "↑"
            elif trend == -1:
                return "↓"
            else:
                return "→"
        else:
            return "→"

    def _get_state_info(self, current_state: Dict[str, Any]) -> str:
        """Extract additional information about the current state."""
        # Placeholder - replace with actual logic to extract relevant state information
        info = ""
        if current_state.get('system_state', 0) == 1:
            info = "Degraded performance detected."
        elif current_state.get('system_state', 0) == 2:
            info = "Critical condition!"
        return info

    def _calculate_correlations(self):
        """Calculate correlations between sensors."""
        for s1 in self.sensor_history:
            for s2 in self.sensor_history:
                if s1 < s2:  # Avoid duplicates
                    corr = np.corrcoef(
                        self.sensor_history[s1][-self.max_history:],
                        self.sensor_history[s2][-self.max_history:]
                    )[0, 1]
                    self.sensor_correlations[f"{s1}_{s2}"] = corr

    def _add_correlation_edges(self):
        """Add edges for highly correlated sensors."""
        for pair, corr in self.sensor_correlations.items():
            if abs(corr) > 0.7:  # Strong correlation
                s1, s2 = pair.split('_')
                self.graph.add_edge(
                    f"{s1}_{self.node_counter['sensor'] - 1}",
                    f"{s2}_{self.node_counter['sensor'] - 1}",
                    weight=abs(corr),
                    label=f'corr: {corr:.2f}',
                    style='dotted'
                )

    def _add_metrics_node(self, performance_metrics: Dict[str, float]):
        """Add a node showing key performance metrics."""
        metrics_id = f"metrics_{self.node_counter['state']}"
        self.graph.add_node(metrics_id,
                            type='metrics',
                            label=(f"Metrics\n"
                                   f"Efficiency: {performance_metrics.get('efficiency_index', 0):.2f}\n"
                                   f"Performance: {performance_metrics.get('performance_score', 0):.2f}"),
                            color='lightgray'
                            )
        return metrics_id

    def update_graph(self,
                     current_state: Dict[str, float],
                     insights: List[str],
                     rules: List[Dict],
                     anomalies: Dict[str, Any]) -> None:
        """Update knowledge graph with new information."""
        try:
            # Add temporal aspect
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Add sensor nodes
            sensor_data = {
                'temperature': current_state.get('temperature', 0.0),
                'vibration': current_state.get('vibration', 0.0),
                'pressure': current_state.get('pressure', 0.0)
            }

            # Add sensor nodes and their relationships
            sensor_nodes = {}
            for sensor, value in sensor_data.items():
                trend = self._calculate_trend(sensor, value)
                node_id = f"{sensor}_{self.node_counter['sensor']}"
                self.graph.add_node(node_id,
                                    type='sensor',
                                    value=value,
                                    trend=trend,
                                    label=f"{sensor}\n{value:.2f}\n{trend}",
                                    color=self.node_types['sensor'],
                                    timestamp=timestamp)
                sensor_nodes[sensor] = node_id
                self.node_counter['sensor'] += 1

            # Add state node
            state_id = f"state_{self.node_counter['state']}"
            state_value = int(current_state.get('system_state', 0))
            state_labels = ['Normal', 'Degraded', 'Critical']
            state_info = self._get_state_info(current_state)
            self.graph.add_node(state_id,
                                type='state',
                                value=state_value,
                                info=state_info,
                                label=f"State\n{state_labels[state_value]}\n{state_info}",
                                color=self.node_types['state'])
            self.node_counter['state'] += 1

            # Add relationships between sensors and state
            for sensor_id in sensor_nodes.values():
                self.graph.add_edge(sensor_id, state_id,
                                    weight=1.0,
                                    label='influences',
                                    timestamp=timestamp)

            # Add rule nodes and their relationships
            rule_nodes = []
            for rule in rules:
                if rule['confidence'] > 0.7:  # Only show high confidence rules
                    rule_id = f"rule_{self.node_counter['rule']}"
                    self.graph.add_node(rule_id,
                                        type='rule',
                                        confidence=rule['confidence'],
                                        label=f"Rule\n{rule['rule'][:30]}...\n{rule['confidence']:.2f}",
                                        color=self.node_types['rule'],
                                        timestamp=timestamp)
                    rule_nodes.append(rule_id)
                    self.node_counter['rule'] += 1

                    # Connect rules to relevant sensors
                    for sensor, sensor_id in sensor_nodes.items():
                        if sensor in rule['rule'].lower():
                            self.graph.add_edge(sensor_id, rule_id,
                                                weight=rule['confidence'],
                                                label='triggers',
                                                timestamp=timestamp)

            # Add anomaly nodes if any anomalies detected
            if anomalies.get('severity', 0) > 0:
                anomaly_id = f"anomaly_{self.node_counter['anomaly']}"
                self.graph.add_node(anomaly_id,
                                    type='anomaly',
                                    severity=anomalies['severity'],
                                    label=f"Anomaly\nSeverity: {anomalies['severity']}",
                                    color=self.node_types['anomaly'],
                                    timestamp=timestamp)
                self.node_counter['anomaly'] += 1

                # Connect anomalies to states and rules
                self.graph.add_edge(state_id, anomaly_id,
                                    weight=1.0,
                                    label='indicates',
                                    timestamp=timestamp)
                for rule_id in rule_nodes:
                    self.graph.add_edge(rule_id, anomaly_id,
                                        weight=self.graph.nodes[rule_id]['confidence'],
                                        label='detects',
                                        timestamp=timestamp)

            # Add insight nodes
            for idx, insight in enumerate(insights):
                insight_id = f"insight_{self.node_counter['insight']}"
                self.graph.add_node(insight_id,
                                    type='insight',
                                    label=f"Insight\n{insight[:50]}...",
                                    color=self.node_types['insight'],
                                    timestamp=timestamp)
                self.node_counter['insight'] += 1

                # Connect insights to anomalies and states
                if anomalies.get('severity', 0) > 0:
                    self.graph.add_edge(anomaly_id, insight_id,
                                        weight=1.0,
                                        label='explains',
                                        timestamp=timestamp)
                self.graph.add_edge(state_id, insight_id,
                                    weight=1.0,
                                    label='generates',
                                    timestamp=timestamp)

            # Add metrics node
            metrics_id = self._add_metrics_node({
                'efficiency_index': current_state.get('efficiency_index', 0),
                'performance_score': current_state.get('performance_score', 0)
            })
            self.graph.add_edge(state_id, metrics_id,
                                weight=1.0,
                                label='has',
                                timestamp=timestamp)

            # Calculate and add correlation edges
            self._calculate_correlations()
            self._add_correlation_edges()

            # Prune old nodes to maintain graph size
            self._prune_old_nodes()

            self.logger.info(f"Knowledge graph updated: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {e}")
            raise

    def _prune_old_nodes(self, max_nodes: int = 100) -> None:
        """Remove oldest nodes if graph exceeds size limit."""
        try:
            if len(self.graph) > max_nodes:
                nodes = sorted(self.graph.nodes(data=True),
                               key=lambda x: x[1].get('timestamp', ''))
                nodes_to_remove = nodes[:(len(self.graph) - max_nodes)]
                self.graph.remove_nodes_from([n[0] for n in nodes_to_remove])
                self.logger.info(f"Pruned {len(nodes_to_remove)} old nodes from graph")
        except Exception as e:
            self.logger.error(f"Error pruning nodes: {e}")

    def visualize(self, output_path: str) -> None:
        """Generate and save knowledge graph visualization."""
        try:
            self.logger.info(f"Starting visualization with {len(self.graph.nodes)} nodes")
            self.logger.info(f"Node types present: {set(nx.get_node_attributes(self.graph, 'type').values())}")

            plt.figure(figsize=(20, 15))  # Larger figure

            # Use spring layout with parameters
            pos = nx.spring_layout(self.graph, **self.layout_params)

            # Draw nodes by type with different sizes
            sizes = {
                'sensor': 2000,
                'state': 3000,
                'rule': 2500,
                'insight': 2000,
                'anomaly': 2500,
                'metrics': 2000  # Size for metrics node
            }

            # Add node clustering
            for node_type in self.node_types:
                nodes = [n for n, d in self.graph.nodes(data=True)
                         if d.get('type') == node_type]
                if nodes:
                    nx.draw_networkx_nodes(self.graph, pos,
                                           nodelist=nodes,
                                           node_color=self.node_types[node_type],
                                           node_size=sizes[node_type],
                                           alpha=0.8,
                                           label=node_type.capitalize())

            # Add better edge styling
            edge_styles = {
                'influences': {'style': 'solid', 'weight': 2},
                'triggers': {'style': 'dashed', 'weight': 1},
                'generates': {'style': 'dotted', 'weight': 1},
                'has': {'style': 'solid', 'weight': 1}  # Style for metrics edge
            }
            for u, v, data in self.graph.edges(data=True):
                style = edge_styles.get(data.get('label'), {}).get('style', 'solid')
                weight = edge_styles.get(data.get('label'), {}).get('weight', 1)
                nx.draw_networkx_edges(self.graph, pos,
                                       edgelist=[(u, v)],
                                       style=style,
                                       width=weight,
                                       edge_color='gray',
                                       arrows=True,
                                       arrowsize=20,
                                       alpha=0.5)

            # Add labels
            labels = nx.get_node_attributes(self.graph, 'label')
            nx.draw_networkx_labels(self.graph, pos, labels,
                                    font_size=8,
                                    font_weight='bold')

            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            nx.draw_networkx_edge_labels(self.graph, pos,
                                         edge_labels=edge_labels,
                                         font_size=6)

            plt.title("NEXUS-DT Knowledge Graph", pad=20, size=16)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.axis('off')
            plt.tight_layout()

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Knowledge graph visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")
            self.logger.error(f"Graph state: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            raise

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current graph state."""
        try:
            stats = {
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges),
                'node_types': {
                    node_type: len([n for n, d in self.graph.nodes(data=True)
                                    if d.get('type') == node_type])
                    for node_type in self.node_types
                },
                'avg_degree': np.mean([d for n, d in self.graph.degree()]),
                'timestamp': datetime.now().isoformat()
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating graph statistics: {e}")
            return {}

    def export_graph(self, output_path: str) -> None:
        """Export graph data to JSON format."""
        try:
            import json
            import networkx as nx

            # Convert graph to JSON-serializable format
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        **{k: str(v) if isinstance(v, datetime) else v
                           for k, v in data.items()}
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        **{k: str(v) if isinstance(v, datetime) else v
                           for k, v in data.items()}
                    }
                    for u, v, data in self.graph.edges(data=True)
                ],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.get_graph_statistics()
                }
            }

            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)

            self.logger.info(f"Graph data exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}")
            raise

