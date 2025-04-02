% Dynamically learned rules for ANSR-DT - Automatically managed
% Last updated: 2025-04-02T06:58:53.672818

:- discontiguous(neural_rule/0).
:- discontiguous(gradient_rule/0).
:- discontiguous(pattern_rule/0).
:- discontiguous(abstract_pattern/0).

neural_rule_1 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.888, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_10 :- feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.999, Extracted: 2025-04-02T06:58:53.672671, Activations: 0
neural_rule_11 :- feature_gradient(efficiency_index, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.920, Extracted: 2025-04-02T06:58:53.672671, Activations: 0
neural_rule_12 :- feature_gradient(efficiency_index, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.785, Extracted: 2025-04-02T06:58:53.672671, Activations: 0
neural_rule_2 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.897, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_3 :- feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.999, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_4 :- feature_gradient(efficiency_index, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.920, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_5 :- feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.996, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_6 :- feature_gradient(efficiency_index, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.895, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_7 :- feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.828, Extracted: 2025-04-02T06:53:22.573991, Activations: 0
neural_rule_8 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.824, Extracted: 2025-04-02T06:58:53.672671, Activations: 0
neural_rule_9 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.897, Extracted: 2025-04-02T06:58:53.672671, Activations: 0
