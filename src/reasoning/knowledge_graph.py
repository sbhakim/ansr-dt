# src/reasoning/knowledge_graph.py
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import os
import numpy as np
import math

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
                     return None # Represent NaN as null in JSON
                 if np.isinf(obj):
                     # Represent Inf as a large number string or null, null is safer
                     return None # Or use str(obj) if compatibility allows
                 return float(obj)
            elif isinstance(obj, np.ndarray):
                # Ensure NaNs/Infs in arrays are handled if needed, e.g., replace with None
                processed_list = [None if isinstance(item, float) and (math.isnan(item) or math.isinf(item)) else item for item in obj.tolist()]
                return processed_list
            # Handle datetime objects
            elif isinstance(obj, datetime):
                 return obj.isoformat()
            return super(NumpyEncoder, self).default(obj)

# Default values for graph parameters
DEFAULT_MAX_NODES = 500  # Less relevant for snapshot approach
DEFAULT_MAX_HISTORY = 1000 # For trend/correlation calculation history

class KnowledgeGraphGenerator:
    """Generates and visualizes a knowledge graph representing a snapshot of
       the current system state, insights, and rules. Includes options for
       generating detailed ('full') and publication-focused ('focused_ieee') plots."""

    def __init__(self, logger: Optional[logging.Logger] = None, max_nodes: int = DEFAULT_MAX_NODES, max_history: int = DEFAULT_MAX_HISTORY):
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        self.max_nodes = max_nodes # Limit kept for potential future use
        self.max_history = max_history # History for trend/correlation calculation

        # Parameterize correlation settings
        self.correlation_threshold = 0.7
        self.min_history_for_correlation = 10

        # Node styling
        self.node_types = {
            'sensor': 'lightblue',
            'state': 'lightgreen',
            'rule': 'lightpink',
            'anomaly': 'salmon',
            'insight': 'lightyellow',
            'metrics': 'lightgray'
        }
        # Shared counter for generating unique node IDs *within a snapshot*
        self.global_node_counter = 0

        # Data storage for trends and correlations (maintained across updates)
        self.sensor_history: Dict[str, List[float]] = {}
        self.sensor_correlations: Dict[str, float] = {}

        # Layout parameters for visualization
        self.layout_params = {
            'k': 0.15, # Default k for spring layout (often needs tuning)
            'iterations': 100,
            'seed': 42,
            # --- NEW: Default Graphviz spacing attributes ---
            'nodesep': '0.6', # Separation between nodes in same rank (Graphviz)
            'ranksep': '1.0', # Separation between ranks/layers (Graphviz)
            'rankdir': 'LR'   # Layout direction (Left-to-Right) (Graphviz)
        }
        # Caching layout
        self._cached_pos: Optional[Dict[Any, Any]] = None
        self._graph_hash: Optional[str] = None

        self.logger.info(f"KnowledgeGraphGenerator initialized (Snapshot Mode) with max_history={self.max_history}")

    def _get_unique_node_id(self, prefix: str) -> str:
        """Generates a unique node ID based on a prefix and a counter for the current snapshot."""
        self.global_node_counter += 1
        return f"{prefix}_{self.global_node_counter}"

    def _calculate_trend(self, sensor: str, value: float) -> str:
        """Calculates the trend (↑, ↓, →, N/A) of a sensor value based on history."""
        if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
            self.logger.debug(f"Invalid value '{value}' for sensor '{sensor}' trend calculation.")
            return "N/A"

        if sensor not in self.sensor_history:
            self.sensor_history[sensor] = []

        self.sensor_history[sensor].append(value)
        if len(self.sensor_history[sensor]) > self.max_history:
             self.sensor_history[sensor] = self.sensor_history[sensor][-self.max_history:]

        if len(self.sensor_history[sensor]) >= 2:
            last_val, prev_val = self.sensor_history[sensor][-1], self.sensor_history[sensor][-2]
            try:
                if math.isnan(last_val) or math.isnan(prev_val) or math.isinf(last_val) or math.isinf(prev_val):
                    return "N/A"
                diff = last_val - prev_val
                tolerance = 1e-6
                if diff > tolerance: return "↑"
                elif diff < -tolerance: return "↓"
                else: return "→"
            except TypeError:
                 self.logger.warning(f"Could not compare values for trend calculation: {last_val}, {prev_val}")
                 return "N/A"
        else:
            return "→"

    def _get_state_info(self, current_state_dict: Dict[str, Any]) -> Tuple[int, str]:
        """Safely extracts system state value (int) and description string."""
        state_value = 0
        state_description = "Normal"
        raw_state = current_state_dict.get('system_state')
        try:
            state_value_float = float(raw_state)
            if math.isnan(state_value_float) or math.isinf(state_value_float):
                 raise ValueError("State value is NaN or Inf")
            state_value = int(round(state_value_float))
            if state_value == 1: state_description = "Degraded"
            elif state_value == 2: state_description = "Critical"
            elif state_value == 0: state_description = "Normal"
            else:
                self.logger.warning(f"Unknown system_state: {raw_state} -> {state_value}. Treating as Normal.")
                state_description = f"Unknown ({state_value})"
                state_value = 0
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Invalid system_state '{raw_state}': {e}. Defaulting to Normal.")
            state_value = 0
            state_description = "Normal (Invalid Input)"
        if state_value not in [0, 1, 2]:
            self.logger.error(f"Corrected invalid state index {state_value} to 0.")
            state_value = 0
            state_description += " [Corrected]"
        return state_value, state_description

    def _calculate_correlations(self):
        """Safely calculate correlations between sensors with sufficient history."""
        sensors = list(self.sensor_history.keys())
        if len(sensors) < 2: return

        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                s1, s2 = sensors[i], sensors[j]
                hist1, hist2 = self.sensor_history.get(s1, []), self.sensor_history.get(s2, [])
                pair_key = f"{s1}_{s2}"
                common_len = min(len(hist1), len(hist2))
                if common_len < self.min_history_for_correlation:
                     self.sensor_correlations.pop(pair_key, None)
                     continue
                try:
                    arr1 = np.array(hist1[-common_len:], dtype=float)
                    arr2 = np.array(hist2[-common_len:], dtype=float)
                    if np.isnan(arr1).any() or np.isnan(arr2).any() or \
                       np.isinf(arr1).any() or np.isinf(arr2).any() or \
                       np.std(arr1) < 1e-9 or np.std(arr2) < 1e-9:
                        self.sensor_correlations.pop(pair_key, None)
                        continue
                    corr_matrix = np.corrcoef(arr1, arr2)
                    if corr_matrix.shape == (2, 2):
                        corr = corr_matrix[0, 1]
                        if not math.isnan(corr): self.sensor_correlations[pair_key] = corr
                        else: self.sensor_correlations.pop(pair_key, None)
                    else: self.sensor_correlations.pop(pair_key, None)
                except Exception as corr_err:
                    self.logger.warning(f"Corr calc error ({s1},{s2}): {corr_err}", exc_info=False)
                    self.sensor_correlations.pop(pair_key, None)

    def _add_edge_safe(self, u_node: Optional[str], v_node: Optional[str], **attrs):
        """Adds an edge only if both source and target nodes exist and are not None."""
        if u_node is None or v_node is None:
             self.logger.debug(f"Skipping edge: Source/target None (u={u_node}, v={v_node}).")
             return
        if self.graph.has_node(u_node) and self.graph.has_node(v_node):
            self.graph.add_edge(u_node, v_node, **attrs)
        else:
            missing = []
            if not self.graph.has_node(u_node): missing.append(f"source '{u_node}'")
            if not self.graph.has_node(v_node): missing.append(f"target '{v_node}'")
            self.logger.warning(f"Skipping edge ({u_node} -> {v_node}): {', '.join(missing)} not found.")

    def update_graph(self,
                     current_state: Dict[str, Any],
                     insights: List[str],
                     rules: List[Dict],
                     anomalies: Dict[str, Any]) -> None:
        """Update knowledge graph as a snapshot of the current state."""
        self.graph.clear()
        self.global_node_counter = 0
        self.logger.debug("Graph cleared for snapshot update.")

        # --- Validate/default current_state ---
        required_keys = ['temperature', 'vibration', 'pressure', 'system_state', 'efficiency_index', 'performance_score']
        defaults = {'temperature': 70.0, 'vibration': 50.0, 'pressure': 30.0, 'system_state': 0, 'efficiency_index': 0.9, 'performance_score': 90.0}
        valid_current_state = {}
        missing_or_invalid = []
        for key in required_keys:
            value = current_state.get(key)
            is_invalid = (value is None or
                          (isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value))) or
                          (isinstance(value, (int, float, np.number)) and not (-1e9 < value < 1e9)))
            if is_invalid:
                 missing_or_invalid.append(key)
                 valid_current_state[key] = defaults[key]
            else:
                 valid_current_state[key] = value
        if missing_or_invalid:
             self.logger.warning(f"Invalid/missing state keys: {missing_or_invalid}. Used defaults.")
        current_state = valid_current_state

        try:
            timestamp_dt = datetime.now()
            timestamp_str = timestamp_dt.isoformat()
            sensor_node_ids: Dict[str, str] = {}
            rule_node_ids: List[str] = []
            insight_node_ids: List[str] = []
            state_id: Optional[str] = None
            metrics_id: Optional[str] = None
            anomaly_id: Optional[str] = None

            # --- 1. Add Nodes ---
            # Sensors
            for sensor in ['temperature', 'vibration', 'pressure']:
                try:
                    value_f = float(current_state[sensor])
                    trend = self._calculate_trend(sensor, value_f)
                    node_id = self._get_unique_node_id(sensor)
                    self.graph.add_node(node_id, type='sensor', sensor_name=sensor, value=value_f, trend=trend,
                                        label=f"{sensor.capitalize()}\n{value_f:.1f} {trend}",
                                        color=self.node_types['sensor'], timestamp=timestamp_dt)
                    sensor_node_ids[sensor] = node_id
                except Exception as e: self.logger.warning(f"Skip sensor node '{sensor}': {e}")
            # State
            state_value, state_label_part = self._get_state_info(current_state)
            state_id = self._get_unique_node_id("state")
            state_labels = ['Normal', 'Degraded', 'Critical']
            self.graph.add_node(state_id, type='state', value=state_value,
                                label=f"State: {state_labels[state_value]}\n({state_label_part})",
                                color=self.node_types['state'], timestamp=timestamp_dt)
            # Metrics
            try:
                eff_f, perf_f = float(current_state['efficiency_index']), float(current_state['performance_score'])
                metrics_id = self._get_unique_node_id("metrics")
                self.graph.add_node(metrics_id, type='metrics', efficiency=eff_f, performance=perf_f,
                                    label=f"Metrics\nEff: {eff_f:.2f}\nPerf: {perf_f:.1f}",
                                    color=self.node_types['metrics'], timestamp=timestamp_dt)
            except Exception as e: self.logger.error(f"Failed add metrics node: {e}")
            # Rules
            for rule_info in rules:
                if isinstance(rule_info, dict) and 'rule' in rule_info and 'confidence' in rule_info:
                    try:
                        conf = float(rule_info['confidence'])
                        if not (0.0 <= conf <= 1.0): raise ValueError("Conf range")
                        rid = self._get_unique_node_id("rule")
                        rstr = rule_info['rule']
                        rhead = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:", rstr)
                        rlabel = rhead.group(1) if rhead else (rstr[:20] + '...')
                        self.graph.add_node(rid, type='rule', rule_string=rstr, confidence=conf,
                                            label=f"Rule: {rlabel}\nConf: {conf:.2f}",
                                            color=self.node_types['rule'], timestamp=timestamp_dt)
                        rule_node_ids.append(rid)
                    except Exception as e: self.logger.warning(f"Skip rule node {rule_info.get('rule')}: {e}")
            # Anomaly
            try:
                 sev = int(anomalies.get('severity', 0))
                 if sev > 0:
                     conf = float(anomalies.get('confidence', 0.0))
                     if not (0.0 <= conf <= 1.0): conf = 0.0
                     anomaly_id = self._get_unique_node_id("anomaly")
                     self.graph.add_node(anomaly_id, type='anomaly', severity=sev, confidence=conf,
                                         label=f"Anomaly\nSev: {sev} Conf: {conf:.2f}",
                                         color=self.node_types['anomaly'], timestamp=timestamp_dt)
            except Exception as e: self.logger.warning(f"Skip anomaly node {anomalies}: {e}")
            # Insights
            for insight_str in insights:
                 if isinstance(insight_str, str) and insight_str.strip():
                    iid = self._get_unique_node_id("insight")
                    ilabel = (insight_str[:47] + '...') if len(insight_str) > 50 else insight_str
                    self.graph.add_node(iid, type='insight', text=insight_str,
                                        label=f"Insight\n{ilabel}",
                                        color=self.node_types['insight'], timestamp=timestamp_dt)
                    insight_node_ids.append(iid)

            # --- 2. Add Edges ---
            edge_ts = timestamp_str
            for sid in sensor_node_ids.values(): self._add_edge_safe(sid, state_id, weight=0.8, label='influences', timestamp=edge_ts)
            self._add_edge_safe(state_id, metrics_id, weight=1.0, label='has', timestamp=edge_ts)
            for rid in rule_node_ids:
                 rdata = self.graph.nodes[rid]; rstr_low = rdata.get('rule_string', '').lower(); rconf = rdata.get('confidence', 0.0)
                 for sname, sid in sensor_node_ids.items():
                     pat = r'\b' + re.escape(sname) + r'\b'; pat_pred = r'\b' + re.escape(sname) + r'\s*\('
                     if re.search(pat, rstr_low) or re.search(pat_pred, rstr_low): self._add_edge_safe(sid, rid, weight=rconf, label='related_to', timestamp=edge_ts)
                 rhead = rstr_low.split(":-")[0].strip()
                 if "degraded" in rhead: self._add_edge_safe(rid, state_id, weight=rconf, label='implies_degraded', timestamp=edge_ts)
                 elif "critical" in rhead: self._add_edge_safe(rid, state_id, weight=rconf, label='implies_critical', timestamp=edge_ts)
            if anomaly_id:
                 self._add_edge_safe(state_id, anomaly_id, weight=1.0, label='indicates', timestamp=edge_ts)
                 for rid in rule_node_ids: self._add_edge_safe(rid, anomaly_id, weight=self.graph.nodes[rid].get('confidence', 0.0), label='detects', timestamp=edge_ts)
            for iid in insight_node_ids:
                 self._add_edge_safe(state_id, iid, weight=0.5, label='context_for', timestamp=edge_ts)
                 if anomaly_id: self._add_edge_safe(anomaly_id, iid, weight=0.8, label='explained_by', timestamp=edge_ts)
                 itext_low = self.graph.nodes[iid].get('text', '').lower()
                 for rid in rule_node_ids:
                      rhead = self.graph.nodes[rid].get('rule_string', '').split(":-")[0].strip().lower()
                      if rhead and rhead in itext_low: self._add_edge_safe(rid, iid, weight=0.9, label='generates', timestamp=edge_ts)
            self._calculate_correlations()
            for pair, corr in self.sensor_correlations.items():
                 if abs(corr) > self.correlation_threshold:
                     try:
                          s1n, s2n = pair.split('_'); s1id, s2id = sensor_node_ids.get(s1n), sensor_node_ids.get(s2n)
                          if s1id and s2id: self._add_edge_safe(s1id, s2id, weight=abs(corr), label=f'corr: {corr:.2f}', style='dotted', timestamp=edge_ts, color='blue')
                     except ValueError: self.logger.warning(f"Parse corr key failed: {pair}")

            self.logger.info(f"KG snapshot populated: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            self.logger.error(f"Critical error constructing KG snapshot: {e}", exc_info=True)
            self.graph.clear()


    def _create_focused_subgraph(self,
                                 rule_confidence_threshold: float = 0.8,
                                 max_rules_to_show: int = 5) -> nx.DiGraph: # Added max_rules param
        """Creates a filtered subgraph for focused visualization."""
        subgraph = nx.DiGraph()
        nodes_to_add = set()
        if not self.graph: return subgraph

        # 1. Core nodes
        state_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'state']
        anomaly_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'anomaly']
        metrics_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'metrics']
        nodes_to_add.update(state_nodes); nodes_to_add.update(anomaly_nodes); nodes_to_add.update(metrics_nodes)

        # 2. High-impact rules (confidence OR state implication)
        high_impact_rules_candidates = set()
        critical_state_node = state_nodes[0] if state_nodes else None
        for rule_node, rule_data in self.graph.nodes(data=True):
             if rule_data.get('type') == 'rule':
                 is_high_conf = rule_data.get('confidence', 0.0) >= rule_confidence_threshold
                 implies_state = False
                 if critical_state_node and self.graph.has_edge(rule_node, critical_state_node):
                      edge_label = self.graph.edges[rule_node, critical_state_node].get('label', '')
                      if 'implies_critical' in edge_label or 'implies_degraded' in edge_label: implies_state = True
                 if is_high_conf or implies_state: high_impact_rules_candidates.add(rule_node)

        # --- Filter rules by max_rules_to_show based on confidence ---
        high_impact_rules = set()
        if len(high_impact_rules_candidates) > max_rules_to_show:
            rules_with_conf = [(node_id, self.graph.nodes[node_id].get('confidence', 0.0)) for node_id in high_impact_rules_candidates]
            rules_with_conf.sort(key=lambda item: item[1], reverse=True) # Sort descending by confidence
            high_impact_rules = set(item[0] for item in rules_with_conf[:max_rules_to_show])
            self.logger.info(f"Filtered focused graph to top {max_rules_to_show} rules by confidence.")
        else:
             high_impact_rules = high_impact_rules_candidates # Keep all candidates if below limit
        nodes_to_add.update(high_impact_rules)

        # 3. Related Sensors (connected to the selected high-impact rules)
        related_sensors = set()
        for rule_node in high_impact_rules: # Use the filtered set
             for pred in self.graph.predecessors(rule_node):
                 if self.graph.has_node(pred) and self.graph.nodes[pred].get('type') == 'sensor':
                     related_sensors.add(pred)
        nodes_to_add.update(related_sensors) # Add the sensors connected to the chosen rules

        # 4. Related Insights (connected to core nodes or selected high-impact rules)
        related_insights = set()
        nodes_for_insight_context = set(state_nodes) | set(anomaly_nodes) | set(high_impact_rules) # Use filtered rules
        for context_node in nodes_for_insight_context:
             if not self.graph.has_node(context_node): continue # Safety check
             # Successors (insights explained by context)
             for succ in self.graph.successors(context_node):
                  if self.graph.has_node(succ) and self.graph.nodes[succ].get('type') == 'insight': related_insights.add(succ)
             # Predecessors (insights generating rule - less common)
             if self.graph.nodes[context_node].get('type') == 'rule':
                 for pred in self.graph.predecessors(context_node):
                     if self.graph.has_node(pred) and self.graph.nodes[pred].get('type') == 'insight' and self.graph.has_edge(pred, context_node):
                          if self.graph.edges[pred, context_node].get('label') == 'generates': related_insights.add(pred)
        nodes_to_add.update(related_insights)

        # 5. Add nodes to subgraph
        for node_id in nodes_to_add:
            if self.graph.has_node(node_id): subgraph.add_node(node_id, **self.graph.nodes[node_id])

        # 6. Add edges between selected nodes
        for u, v, data in self.graph.edges(data=True):
            if u in nodes_to_add and v in nodes_to_add:
                is_correlation = data.get('label', '').startswith('corr:')
                correlation_weight_threshold = 0.75 # Stricter for focused
                if is_correlation and abs(data.get('weight', 0.0)) < correlation_weight_threshold: continue
                subgraph.add_edge(u, v, **data)

        self.logger.info(f"Created focused subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        return subgraph


    def visualize(self,
                  output_path: str,
                  plot_type: str = 'full', # 'full' or 'focused_ieee'
                  layout_engine: str = 'spring', # 'spring', 'kamada_kawai', 'dot', 'neato', etc.
                  rule_confidence_threshold: float = 0.8, # For 'focused_ieee'
                  max_rules_to_show_focused: int = 5, # Limit rules in focused plot
                  show_edge_labels: bool = True, # Control edge label visibility
                  font_scale: float = 1.0, # Scale font sizes
                  use_cache: bool = True, # Use layout cache
                  dpi: int = 300, # Resolution for paper
                  # Graphviz layout parameters (optional override)
                  gv_nodesep: Optional[str] = None, # e.g., '0.6'
                  gv_ranksep: Optional[str] = None, # e.g., '1.0'
                  gv_rankdir: Optional[str] = None  # e.g., 'LR'
                  ) -> None:
        """
        Generates and saves knowledge graph visualization with customizable options.
        """

        # --- Determine target graph and specific settings ---
        if plot_type == 'focused_ieee':
            target_graph = self._create_focused_subgraph(rule_confidence_threshold, max_rules_to_show_focused)
            effective_show_edge_labels = False
            effective_font_scale = font_scale * 0.85
            layout_engine = 'dot' if layout_engine == 'spring' else layout_engine # Default 'dot' for IEEE
            # Use specific Graphviz params if provided, else use defaults from self.layout_params
            gv_nodesep = gv_nodesep or self.layout_params.get('nodesep', '0.6')
            gv_ranksep = gv_ranksep or self.layout_params.get('ranksep', '1.0')
            gv_rankdir = gv_rankdir or self.layout_params.get('rankdir', 'LR')
        else: # 'full' plot
            target_graph = self.graph
            effective_show_edge_labels = show_edge_labels
            effective_font_scale = font_scale
            # Use potentially different defaults for full plot if Graphviz is used
            gv_nodesep = gv_nodesep or self.layout_params.get('nodesep', '0.4')
            gv_ranksep = gv_ranksep or self.layout_params.get('ranksep', '0.8')
            gv_rankdir = gv_rankdir or self.layout_params.get('rankdir', 'TB') # TB often better for larger graphs

        if not target_graph or target_graph.number_of_nodes() == 0:
            self.logger.warning(f"Target graph ('{plot_type}') empty, skipping visualization: {output_path}.")
            return

        try:
            num_nodes = target_graph.number_of_nodes()
            num_edges = target_graph.number_of_edges()
            self.logger.info(f"Visualizing '{plot_type}' graph ({output_path}): {num_nodes} nodes, {num_edges} edges.")

            # --- Figure Size ---
            figsize_base = 10 if plot_type == 'focused_ieee' else 15
            scale_factor = math.log10(max(10, num_nodes)) / 1.5 if num_nodes > 0 else 1.0
            figsize_scale = min(max(0.7 if plot_type == 'focused_ieee' else 1.0, scale_factor), 2.5)
            fig_width = max(8, min(30, figsize_base * figsize_scale))
            fig_height = max(6, min(25, fig_width * 0.7))
            plt.figure(figsize=(fig_width, fig_height))

            # --- Layout Calculation ---
            pos = None
            structure_hash_elements = (tuple(sorted(target_graph.nodes())), tuple(sorted(target_graph.edges())))
            current_target_graph_hash = hash(structure_hash_elements)
            cache_valid = (use_cache and self._cached_pos is not None and
                           self._graph_hash == current_target_graph_hash and
                           len(self._cached_pos) == num_nodes and
                           set(self._cached_pos.keys()) == set(target_graph.nodes()))

            if cache_valid:
                pos = self._cached_pos
                self.logger.debug("Using cached layout positions.")
            else:
                self.logger.info(f"Calculating '{layout_engine}' layout for '{plot_type}' plot...")
                self._cached_pos = None; self._graph_hash = None # Clear cache
                try:
                    if layout_engine == 'spring':
                         effective_k = 1.2 / math.sqrt(num_nodes) if num_nodes > 1 else 0.8
                         pos = nx.spring_layout(target_graph, k=effective_k, iterations=self.layout_params['iterations'], seed=self.layout_params['seed'])
                    elif layout_engine == 'kamada_kawai':
                         pos = nx.kamada_kawai_layout(target_graph)
                    elif layout_engine in ['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo']:
                         graphviz_args = f"-Gnodesep={gv_nodesep} -Granksep={gv_ranksep} -Grankdir={gv_rankdir} -Gsplines=true -Goverlap=false"
                         try:
                             import pygraphviz
                             pos = nx.nx_agraph.graphviz_layout(target_graph, prog=layout_engine, args=graphviz_args)
                             self.logger.info(f"Used pygraphviz for '{layout_engine}' layout with args: {graphviz_args.replace('-G','')}")
                         except ImportError:
                             try:
                                 import pydot
                                 pos = nx.nx_pydot.graphviz_layout(target_graph, prog=layout_engine, args=graphviz_args)
                                 self.logger.info(f"Used pydot for '{layout_engine}' layout with args: {graphviz_args.replace('-G','')}")
                             except ImportError:
                                 self.logger.error(f"Layout '{layout_engine}' requires Graphviz & pygraphviz/pydot. Falling back to spring.")
                                 pos = nx.spring_layout(target_graph, k=1.2/math.sqrt(num_nodes) if num_nodes>1 else 0.8, seed=self.layout_params['seed'])
                    else:
                        self.logger.warning(f"Unknown layout '{layout_engine}'. Defaulting to spring.")
                        pos = nx.spring_layout(target_graph, k=1.2/math.sqrt(num_nodes) if num_nodes>1 else 0.8, seed=self.layout_params['seed'])

                    if pos and use_cache:
                        self._cached_pos = pos
                        self._graph_hash = current_target_graph_hash

                except Exception as layout_error:
                    self.logger.error(f"Layout calculation '{layout_engine}' failed: {layout_error}. Falling back to random.", exc_info=True)
                    pos = nx.random_layout(target_graph, seed=self.layout_params['seed'])

            if pos is None:
                 self.logger.error("Layout position is None after attempts. Using random.")
                 pos = nx.random_layout(target_graph, seed=self.layout_params['seed'])

            # --- Node Drawing ---
            if plot_type == 'focused_ieee': base_node_sizes = {'sensor': 1200, 'state': 1800, 'rule': 1000, 'insight': 1100, 'anomaly': 1500, 'metrics': 1200}
            else: base_node_sizes = {'sensor': 1500, 'state': 2000, 'rule': 1200, 'insight': 1300, 'anomaly': 1800, 'metrics': 1500}
            node_size_list = [base_node_sizes.get(data.get('type', ''), 1000) * effective_font_scale for _, data in target_graph.nodes(data=True)]
            node_color_list = [self.node_types.get(data.get('type', ''), 'gray') for _, data in target_graph.nodes(data=True)]
            nx.draw_networkx_nodes(target_graph, pos, node_color=node_color_list, node_size=node_size_list, alpha=0.85,
                                   edgecolors='#555555', linewidths=0.5)

            # --- Edge Drawing ---
            edge_styles = { 'influences': {'style': 'solid', 'width': 1.0, 'color': '#555555'}, 'related_to': {'style': 'dashed', 'width': 0.7, 'color': '#888888'}, 'implies_degraded': {'style': 'solid', 'width': 1.3, 'color': 'orange'}, 'implies_critical': {'style': 'solid', 'width': 1.6, 'color': 'red'}, 'indicates': {'style': 'solid', 'width': 1.3, 'color': '#E57373'}, 'detects': {'style': 'dashed', 'width': 0.9, 'color': '#E57373'}, 'context_for': {'style': 'dotted', 'width': 0.8, 'color': '#81C784'}, 'explained_by': {'style': 'solid', 'width': 0.9, 'color': '#81C784'}, 'generates': {'style': 'dotted', 'width': 0.8, 'color': '#9575CD'}, 'has': {'style': 'solid', 'width': 0.9, 'color': 'black'}, 'corr': {'style': 'dotted', 'width': 1.1, 'color': 'blue'} }
            default_edge_style = {'style': 'solid', 'width': 0.5, 'color': '#CCCCCC'}

            for u, v, data in target_graph.edges(data=True):
                elabel = data.get('label', '')
                blabel = 'corr' if elabel.startswith('corr:') else elabel
                style_attrs = edge_styles.get(blabel, default_edge_style)
                weight = data.get('weight', 1.0)
                try:
                    w_mult = max(0.5, min(2.0, float(weight)))
                except (ValueError, TypeError):
                    w_mult = 1.0
                d_width = max(0.4, style_attrs['width'] * w_mult)
                nx.draw_networkx_edges(target_graph, pos, edgelist=[(u, v)], style=style_attrs['style'],
                                       width=d_width, edge_color=style_attrs['color'], arrows=True,
                                       arrowsize=12, alpha=0.7, connectionstyle='arc3,rad=0.05')

            # --- Label Drawing ---
            base_font_size = 7 if plot_type == 'focused_ieee' else 8
            node_font_size = max(5, base_font_size * effective_font_scale)
            edge_font_size = max(4, (base_font_size - 1.5) * effective_font_scale)

            if plot_type == 'focused_ieee':
                labels = {}
                for n, data in target_graph.nodes(data=True):
                    ntype = data.get('type')
                    if ntype == 'sensor': labels[n] = f"{data.get('sensor_name','S')[:4]}\n{data.get('value',0):.1f}{data.get('trend','')}"
                    elif ntype == 'state': slbls = ['N','D','C']; labels[n] = f"State\n{slbls[data.get('value',0)]}"
                    elif ntype == 'metrics': labels[n] = f"Metrics\nE:{data.get('efficiency',0):.2f} P:{data.get('performance',0):.1f}"
                    elif ntype == 'anomaly': labels[n] = f"Key Event\nS:{data.get('severity',0)} C:{data.get('confidence',0):.1f}"
                    elif ntype == 'rule': rname=data.get('rule_string','R').split(":-")[0].strip().replace('neural_rule_','nr_'); labels[n] = f"{rname}\nC:{data.get('confidence',0):.2f}"
                    elif ntype == 'insight': txt=data.get('text','I'); m=re.search(r': (\w+)',txt); labels[n]=f"Insight\n{m.group(1) if m else txt[:10]}"
                    else: labels[n] = data.get('label', n)[:10]
            else: labels = {n: data.get('label', n).replace('\n',' ')[:30] for n, data in target_graph.nodes(data=True)}

            nx.draw_networkx_labels(target_graph, pos, labels, font_size=node_font_size, font_weight='normal')

            # --- Edge Label Drawing (Corrected) ---
            if effective_show_edge_labels and num_edges < 75:
                edge_labels_dict = {}
                for u, v, data in target_graph.edges(data=True):
                     lbl = data.get('label')
                     if lbl: # Only process if label exists
                          if lbl.startswith('corr:'):
                               # --- FIX: Handle potential conversion errors ---
                               try:
                                   parts = lbl.split(':')
                                   if len(parts) > 1:
                                       cval = float(parts[1]) # cval defined here
                                       # Format and add to dictionary only if conversion succeeded
                                       edge_labels_dict[(u, v)] = f'{cval:.1f}'
                                   else:
                                        # Handle case like "corr:" with no value
                                        self.logger.debug(f"Skipping edge label for malformed correlation: {lbl}")
                               except (ValueError, IndexError) as parse_error:
                                    # Handle cases where split fails or conversion to float fails
                                    self.logger.debug(f"Could not parse correlation value from label '{lbl}': {parse_error}. Skipping edge label.")
                               # --- END FIX ---
                          else:
                               # Shorten other labels
                               edge_labels_dict[(u, v)] = lbl[:15]

                # Draw the labels that were successfully created
                nx.draw_networkx_edge_labels(target_graph, pos, edge_labels=edge_labels_dict,
                                            font_size=edge_font_size, font_color='#333333',
                                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))


            plt.title(f"ANSR-DT KG {'Focused ' if plot_type=='focused_ieee' else ''}Snapshot ({datetime.now().strftime('%Y-%m-%d %H:%M')})", size=node_font_size + 2)
            plt.axis('off')
            try: plt.tight_layout(pad=0.1)
            except Exception as tight_layout_error: self.logger.warning(f"tight_layout failed: {tight_layout_error}")

            # --- Save Figure ---
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
            self.logger.info(f"'{plot_type}' visualization saved to {output_path} (dpi={dpi})")

        except Exception as e:
            self.logger.error(f"Core visualization error ('{plot_type}'): {str(e)}", exc_info=True)
        finally:
             plt.close() # IMPORTANT: Close figure to release memory

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current graph state (snapshot)."""
        try:
            num_nodes = self.graph.number_of_nodes()
            if num_nodes == 0: return {'total_nodes': 0, 'total_edges': 0, 'node_types': {}, 'avg_degree': 0.0, 'timestamp': datetime.now().isoformat()}
            node_types_present = set(nx.get_node_attributes(self.graph, 'type').values())
            node_types_counts = { nt: len([n for n, d in self.graph.nodes(data=True) if d.get('type') == nt]) for nt in node_types_present }
            degrees = [d for n, d in self.graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0.0
            density = nx.density(self.graph) if num_nodes > 1 else 0.0
            stats = { 'total_nodes': num_nodes, 'total_edges': self.graph.number_of_edges(), 'node_types': node_types_counts, 'avg_degree': float(avg_degree), 'density': float(density), 'timestamp': datetime.now().isoformat() }
            return stats
        except Exception as e:
            self.logger.error(f"Error calculating graph stats: {e}", exc_info=True)
            return {'error': str(e)}

    def export_graph(self, output_path: str) -> None:
        """Export graph data to JSON format using networkx node_link_data."""
        try:
            graph_data = nx.node_link_data(self.graph)
            graph_data['metadata'] = { 'export_timestamp': datetime.now().isoformat(), 'statistics': self.get_graph_statistics(), 'source_system': 'ANSR-DT KG' }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2, cls=NumpyEncoder)
            self.logger.info(f"Graph data exported to {output_path}")
        except TypeError as te:
             self.logger.error(f"Serialization error exporting graph: {te}. Check data types.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}", exc_info=True)