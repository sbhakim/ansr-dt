# src/reasoning/knowledge_graph.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Set, Union
import logging
from datetime import datetime
import os
import gzip
import json
from threading import Lock


class KnowledgeGraphGenerator:
    """Knowledge graph generator for NEXUS-DT with enhanced memory management and visualization."""

    def __init__(self, logger: Optional[logging.Logger] = None, max_history: int = 1000):
        """Initialize the knowledge graph generator.

        Args:
            logger: Optional logger instance
            max_history: Maximum history size for tracking
        """
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.max_history = max_history
        self.graph_lock = Lock()

        # Node type definitions with colors
        self.node_types = {
            'sensor': 'lightblue',
            'state': 'lightgreen',
            'rule': 'lightpink',
            'anomaly': 'salmon',
            'insight': 'lightyellow',
            'metrics': 'lightgray'
        }

        # Base node sizes
        self.sizes = {
            'sensor': 2000,
            'state': 3000,
            'rule': 2500,
            'insight': 2000,
            'anomaly': 2500,
            'metrics': 2000
        }

        # Node counters for unique IDs
        self.node_counter = {
            'sensor': 0,
            'state': 0,
            'rule': 0,
            'anomaly': 0,
            'insight': 0,
            'metrics': 0
        }

        # Historical data tracking
        self.sensor_history = {}
        self.sensor_correlations = {}
        self.performance_history = {}

        # Layout parameters
        self.layout_params = {
            'k': 2,
            'iterations': 100,
            'seed': 42
        }

        # Node storage by type for faster lookups
        self.nodes_by_type = {
            node_type: {} for node_type in self.node_types
        }

    def _calculate_trend(self, sensor: str, value: float) -> str:
        """Calculate trend direction for sensor value.

        Args:
            sensor: Sensor name
            value: Current sensor value

        Returns:
            str: Trend indicator (↑, ↓, or →)
        """
        if sensor not in self.sensor_history:
            self.sensor_history[sensor] = []

        self.sensor_history[sensor].append(value)

        if len(self.sensor_history[sensor]) >= 2:
            trend = np.sign(self.sensor_history[sensor][-1] - self.sensor_history[sensor][-2])
            return "↑" if trend == 1 else "↓" if trend == -1 else "→"
        return "→"

    def _calculate_node_importance(self, node_type: str, attrs: Dict[str, Any]) -> float:
        """Calculate node importance for retention decisions.

        Args:
            node_type: Type of the node
            attrs: Node attributes

        Returns:
            float: Importance score
        """
        base_importance = {
            'state': 0.9,
            'rule': 0.8,
            'anomaly': 0.7,
            'sensor': 0.6,
            'insight': 0.5,
            'metrics': 0.4
        }.get(node_type, 0.3)

        confidence = attrs.get('confidence', 0.5)
        severity = attrs.get('severity', 0.0)

        return base_importance * (1 + confidence + severity)

    def _get_node_size(self, node_attrs: Dict) -> float:
        """Calculate node size based on importance and type.

        Args:
            node_attrs: Node attributes

        Returns:
            float: Calculated node size
        """
        base_size = self.sizes.get(node_attrs.get('type', 'default'), 1000)
        importance = node_attrs.get('importance', 0.5)
        confidence = node_attrs.get('confidence', 1.0)
        return base_size * (0.5 + importance * confidence)

    def _calculate_correlations(self):
        """Calculate correlations between sensor values."""
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
            if abs(corr) > 0.7:  # Strong correlation threshold
                s1, s2 = pair.split('_')
                source = f"{s1}_{self.node_counter['sensor'] - 1}"
                target = f"{s2}_{self.node_counter['sensor'] - 1}"

                self.graph.add_edge(
                    source, target,
                    weight=abs(corr),
                    label=f'corr: {corr:.2f}',
                    style='dotted'
                )

    def _get_state_info(self, current_state: Dict[str, Any]) -> str:
        """Get descriptive information about current state.

        Args:
            current_state: Current system state

        Returns:
            str: State description
        """
        state_value = current_state.get('system_state', 0)
        if state_value == 2:
            return "Critical condition!"
        elif state_value == 1:
            return "Degraded performance detected."
        return "Normal operation"

    def _cleanup_node_counters(self):
        """Reset node counters based on actual nodes in graph."""
        with self.graph_lock:
            for node_type in self.node_types:
                self.node_counter[node_type] = len([
                    n for n, d in self.graph.nodes(data=True)
                    if d.get('type') == node_type
                ])

    def _validate_node_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean node attributes.

        Args:
            attrs: Node attributes

        Returns:
            Dict[str, Any]: Cleaned attributes
        """
        return {
            k: v for k, v in attrs.items()
            if not isinstance(v, (datetime, np.ndarray))
        }

    def update_graph(self, current_state: Dict[str, float],
                     insights: List[str],
                     rules: List[Dict],
                     anomalies: Dict[str, Any]) -> None:
        """Update knowledge graph with new information.

        Args:
            current_state: Current system state and readings
            insights: Generated insights
            rules: Active rules
            anomalies: Detected anomalies
        """
        try:
            with self.graph_lock:
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Add sensor nodes
                sensor_data = {
                    'temperature': current_state.get('temperature', 0.0),
                    'vibration': current_state.get('vibration', 0.0),
                    'pressure': current_state.get('pressure', 0.0)
                }

                # Create sensor nodes
                sensor_nodes = {}
                for sensor, value in sensor_data.items():
                    trend = self._calculate_trend(sensor, value)
                    node_id = f"{sensor}_{self.node_counter['sensor']}"

                    # Calculate node importance
                    attrs = {
                        'type': 'sensor',
                        'value': value,
                        'trend': trend,
                        'label': f"{sensor}\n{value:.2f}\n{trend}",
                        'color': self.node_types['sensor'],
                        'timestamp': timestamp
                    }

                    importance = self._calculate_node_importance('sensor', attrs)
                    attrs['importance'] = importance

                    self.graph.add_node(node_id, **attrs)
                    sensor_nodes[sensor] = node_id
                    self.nodes_by_type['sensor'][node_id] = attrs
                    self.node_counter['sensor'] += 1

                # Add state node
                state_id = f"state_{self.node_counter['state']}"
                state_value = int(current_state.get('system_state', 0))
                state_labels = ['Normal', 'Degraded', 'Critical']
                state_info = self._get_state_info(current_state)

                state_attrs = {
                    'type': 'state',
                    'value': state_value,
                    'info': state_info,
                    'label': f"State\n{state_labels[state_value]}\n{state_info}",
                    'color': self.node_types['state'],
                    'importance': 0.9,
                    'timestamp': timestamp
                }

                self.graph.add_node(state_id, **state_attrs)
                self.nodes_by_type['state'][state_id] = state_attrs
                self.node_counter['state'] += 1

                # Connect sensors to state
                for sensor_id in sensor_nodes.values():
                    self.graph.add_edge(sensor_id, state_id,
                                        weight=1.0,
                                        label='influences',
                                        timestamp=timestamp)

                # Add rules with high confidence
                rule_nodes = []
                for rule in rules:
                    if rule['confidence'] > 0.7:
                        rule_id = f"rule_{self.node_counter['rule']}"

                        rule_attrs = {
                            'type': 'rule',
                            'confidence': rule['confidence'],
                            'label': f"Rule\n{rule['rule'][:30]}...\n{rule['confidence']:.2f}",
                            'color': self.node_types['rule'],
                            'importance': 0.8 * rule['confidence'],
                            'timestamp': timestamp
                        }

                        self.graph.add_node(rule_id, **rule_attrs)
                        rule_nodes.append(rule_id)
                        self.nodes_by_type['rule'][rule_id] = rule_attrs
                        self.node_counter['rule'] += 1

                        # Connect rules to related sensors
                        for sensor, sensor_id in sensor_nodes.items():
                            if sensor in rule['rule'].lower():
                                self.graph.add_edge(sensor_id, rule_id,
                                                    weight=rule['confidence'],
                                                    label='triggers',
                                                    timestamp=timestamp)

                # Add anomaly nodes if detected
                if anomalies.get('severity', 0) > 0:
                    anomaly_id = f"anomaly_{self.node_counter['anomaly']}"

                    anomaly_attrs = {
                        'type': 'anomaly',
                        'severity': anomalies['severity'],
                        'label': f"Anomaly\nSeverity: {anomalies['severity']}",
                        'color': self.node_types['anomaly'],
                        'importance': 0.7 * (anomalies['severity'] / 10),
                        'timestamp': timestamp
                    }

                    self.graph.add_node(anomaly_id, **anomaly_attrs)
                    self.nodes_by_type['anomaly'][anomaly_id] = anomaly_attrs
                    self.node_counter['anomaly'] += 1

                    # Connect anomaly to state and rules
                    self.graph.add_edge(state_id, anomaly_id,
                                        weight=1.0,
                                        label='indicates',
                                        timestamp=timestamp)

                    for rule_id in rule_nodes:
                        self.graph.add_edge(rule_id, anomaly_id,
                                            weight=self.graph.nodes[rule_id]['confidence'],
                                            label='detects',
                                            timestamp=timestamp)

                # Add insights
                for insight in insights:
                    insight_id = f"insight_{self.node_counter['insight']}"

                    insight_attrs = {
                        'type': 'insight',
                        'label': f"Insight\n{insight[:50]}...",
                        'color': self.node_types['insight'],
                        'importance': 0.5,
                        'timestamp': timestamp
                    }

                    self.graph.add_node(insight_id, **insight_attrs)
                    self.nodes_by_type['insight'][insight_id] = insight_attrs
                    self.node_counter['insight'] += 1

                    # Connect insights to anomalies and state
                    if 'anomaly_id' in locals():
                        self.graph.add_edge(anomaly_id, insight_id,
                                            weight=1.0,
                                            label='explains',
                                            timestamp=timestamp)

                    self.graph.add_edge(state_id, insight_id,
                                        weight=1.0,
                                        label='generates',
                                        timestamp=timestamp)

                # Update correlations and add correlation edges
                self._calculate_correlations()
                self._add_correlation_edges()

                # Prune if needed
                if len(self.graph) > self.max_history:
                    self._prune_old_nodes()

                self.logger.info(f"Graph updated: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        except Exception as e:
            self.logger.error(f"Error updating graph: {e}")
            raise

    def _prune_old_nodes(self, max_nodes: int = 100) -> None:
        """Remove oldest nodes while preserving important ones."""
        try:
            if len(self.graph) > max_nodes:
                with self.graph_lock:
                    # Keep high-priority nodes
                    critical_nodes = {
                        n for n, d in self.graph.nodes(data=True)
                        if (d.get('type') in ['state', 'rule', 'anomaly'] and
                            d.get('importance', 0) > 0.7)
                    }

                    # Sort remaining nodes by timestamp and importance
                    other_nodes = sorted(
                        [n for n in self.graph.nodes if n not in critical_nodes],
                        key=lambda x: (
                            self.graph.nodes[x].get('timestamp', ''),
                            -self.graph.nodes[x].get('importance', 0)
                        )
                    )

                    # Remove oldest, least important nodes
                    to_remove = len(self.graph) - max_nodes
                    nodes_to_remove = other_nodes[:to_remove]
                    self.graph.remove_nodes_from(nodes_to_remove)

                    # Update counters and storage
                    self._cleanup_node_counters()

                    self.logger.info(f"Pruned {len(nodes_to_remove)} nodes")

        except Exception as e:
            self.logger.error(f"Error pruning nodes: {e}")

    def visualize(self, output_path: str, focus_node: Optional[str] = None) -> None:
        """Generate and save graph visualization.

        Args:
            output_path: Path to save the visualization
            focus_node: Optional node to focus on
        """
        try:
            plt.figure(figsize=(20, 15))

            # Calculate layout
            pos = nx.spring_layout(self.graph, **self.layout_params)
            if focus_node and focus_node in self.graph:
                center_pos = np.array([0.5, 0.5])
                offset = center_pos - pos[focus_node]
                pos = {node: p + offset for node, p in pos.items()}

            # Draw nodes by type
            for node_type, color in self.node_types.items():
                nodes = [n for n in self.nodes_by_type[node_type]]
                if nodes:
                    nx.draw_networkx_nodes(
                        self.graph, pos,
                        nodelist=nodes,
                        node_color=color,
                        node_size=[self._get_node_size(self.graph.nodes[n]) for n in nodes],
                        alpha=0.8,
                        label=node_type.capitalize()
                    )

            # Draw edges
            edge_styles = {
                'influences': {'style': 'solid', 'weight': 2},
                'triggers': {'style': 'dashed', 'weight': 1},
                'generates': {'style': 'dotted', 'weight': 1},
                'has': {'style': 'solid', 'weight': 1}
            }

            for edge_type, style in edge_styles.items():
                edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                         if d.get('label') == edge_type]
                if edges:
                    nx.draw_networkx_edges(
                        self.graph, pos,
                        edgelist=edges,
                        style=style['style'],
                        width=style['weight'],
                        edge_color='gray',
                        alpha=0.5
                    )

            # Add labels
            labels = nx.get_node_attributes(self.graph, 'label')
            nx.draw_networkx_labels(self.graph, pos,
                                    labels=labels,
                                    font_size=8,
                                    font_weight='bold')

            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            nx.draw_networkx_edge_labels(self.graph, pos,
                                         edge_labels=edge_labels,
                                         font_size=6)

            plt.title("NEXUS-DT Knowledge Graph", pad=20, size=16)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.axis('off')
            plt.tight_layout()

            # Save visualization
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                        format='png', transparent=False)
            plt.close()

            self.logger.info(f"Visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Visualization error: {e}")
            raise

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics.

        Returns:
            Dict[str, Any]: Graph statistics
        """
        try:
            with self.graph_lock:
                stats = {
                    'total_nodes': len(self.graph.nodes),
                    'total_edges': len(self.graph.edges),
                    'node_types': {
                        node_type: len(self.nodes_by_type[node_type])
                        for node_type in self.node_types
                    },
                    'avg_degree': float(np.mean([d for n, d in self.graph.degree()])),
                    'timestamp': datetime.now().isoformat()
                }
                return stats
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}

    def export_graph(self, output_path: str) -> None:
        """Export graph data to compressed JSON.

        Args:
            output_path: Path to save the exported data
        """
        try:
            with self.graph_lock:
                graph_data = {
                    'nodes': [{
                        'id': node,
                        **self._validate_node_attrs(data)
                    } for node, data in self.graph.nodes(data=True)],
                    'edges': [{
                        'source': u,
                        'target': v,
                        **self._validate_node_attrs(data)
                    } for u, v, data in self.graph.edges(data=True)],
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'version': '2.0',
                        'statistics': self.get_graph_statistics()
                    }
                }

                # Save compressed
                output_path_gz = f"{output_path}.gz"
                os.makedirs(os.path.dirname(output_path_gz), exist_ok=True)
                with gzip.open(output_path_gz, 'wt', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2)

                self.logger.info(f"Graph exported to {output_path_gz}")

        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise

    @classmethod
    def import_graph(cls, input_path: str) -> 'KnowledgeGraphGenerator':
        """Create knowledge graph from exported data.

        Args:
            input_path: Path to the exported graph data

        Returns:
            KnowledgeGraphGenerator: Initialized graph
        """
        try:
            if input_path.endswith('.gz'):
                with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(input_path) as f:
                    data = json.load(f)

            graph = cls()

            with graph.graph_lock:
                # Reconstruct nodes
                for node in data['nodes']:
                    node_id = node.pop('id')
                    node_type = node.get('type')
                    if node_type:
                        graph.nodes_by_type[node_type][node_id] = node
                    graph.graph.add_node(node_id, **node)

                # Reconstruct edges
                for edge in data['edges']:
                    source = edge.pop('source')
                    target = edge.pop('target')
                    graph.graph.add_edge(source, target, **edge)

                # Update counters
                graph._cleanup_node_counters()

            return graph

        except Exception as e:
            raise ValueError(f"Import failed: {str(e)}")

def __enter__(self):
    """Context manager entry."""
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit with cleanup."""
    if exc_type is not None:
        self.logger.error(f"Error during graph operations: {exc_val}")
    try:
        plt.close('all')  # Close any open plots
    except Exception as e:
        self.logger.warning(f"Error closing plots: {e}")

def __repr__(self) -> str:
    """String representation of the graph."""
    stats = self.get_graph_statistics()
    return (f"KnowledgeGraphGenerator("
            f"nodes={stats['total_nodes']}, "
            f"edges={stats['total_edges']}, "
            f"types={len(self.node_types)})")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("KnowledgeGraph")

    try:
        # Create test data
        current_state = {
            'temperature': 75.5,
            'vibration': 45.2,
            'pressure': 30.1,
            'system_state': 0,
            'efficiency_index': 0.85,
            'performance_score': 90.0
        }

        insights = [
            "Normal operation detected",
            "All sensors within expected ranges"
        ]

        rules = [
            {
                'rule': "temperature > 70 -> monitor",
                'confidence': 0.85
            },
            {
                'rule': "pressure < 35 -> check",
                'confidence': 0.75
            }
        ]

        anomalies = {
            'severity': 0,
            'sensor_anomalies': [],
            'performance_anomalies': []
        }

        # Initialize and use graph
        with KnowledgeGraphGenerator() as kg:
            # Update graph
            kg.update_graph(
                current_state=current_state,
                insights=insights,
                rules=rules,
                anomalies=anomalies
            )

            # Visualize
            kg.visualize('example_graph.png')

            # Export
            kg.export_graph('example_graph.json')

            # Get statistics
            stats = kg.get_graph_statistics()
            logger.info(f"Graph statistics: {stats}")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise