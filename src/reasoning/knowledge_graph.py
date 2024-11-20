# src/reasoning/knowledge_graph.py

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
from datetime import datetime
import os
import numpy as np


class KnowledgeGraphGenerator:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        # Update color definitions - remove # prefix
        self.node_types = {
            'sensor': 'lightblue',
            'state': 'lightgreen',
            'rule': 'lightpink',
            'anomaly': 'salmon',
            'insight': 'lightyellow'
        }
        self.node_counter = {
            'sensor': 0,
            'state': 0,
            'rule': 0,
            'anomaly': 0,
            'insight': 0
        }

    def update_graph(self,
                     current_state: Dict[str, float],
                     insights: List[str],
                     rules: List[Dict],
                     anomalies: Dict[str, Any]) -> None:
        """Update knowledge graph with new information."""
        try:
            timestamp = datetime.now().isoformat()

            # Add sensor nodes
            sensor_data = {
                'temperature': current_state.get('temperature', 0.0),
                'vibration': current_state.get('vibration', 0.0),
                'pressure': current_state.get('pressure', 0.0)
            }

            # Add sensor nodes and their relationships
            sensor_nodes = {}
            for sensor, value in sensor_data.items():
                node_id = f"{sensor}_{self.node_counter['sensor']}"
                self.graph.add_node(node_id,
                                    type='sensor',
                                    value=value,
                                    label=f"{sensor}\n{value:.2f}",
                                    color=self.node_types['sensor'],
                                    timestamp=timestamp)
                sensor_nodes[sensor] = node_id
                self.node_counter['sensor'] += 1

            # Add state node
            state_id = f"state_{self.node_counter['state']}"
            state_value = int(current_state.get('system_state', 0))
            state_labels = ['Normal', 'Degraded', 'Critical']
            self.graph.add_node(state_id,
                                type='state',
                                value=state_value,
                                label=f"State\n{state_labels[state_value]}",
                                color=self.node_types['state'],
                                timestamp=timestamp)
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

            plt.figure(figsize=(15, 10))

            # Create layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)

            # Draw nodes by type
            for node_type, color in self.node_types.items():
                nodes = [n for n, d in self.graph.nodes(data=True)
                         if d.get('type') == node_type]
                if nodes:
                    nx.draw_networkx_nodes(self.graph, pos,
                                           nodelist=nodes,
                                           node_color=color,
                                           node_size=2000,
                                           alpha=0.8,
                                           label=node_type.capitalize())

            # Draw edges
            nx.draw_networkx_edges(self.graph, pos,
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
                                         edge_labels,
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