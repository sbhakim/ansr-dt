% Dynamically learned rules for ANSR-DT - Automatically managed
% Last updated: 2025-04-10T08:03:53.244530

:- discontiguous(neural_rule/0).
:- discontiguous(gradient_rule/0).
:- discontiguous(pattern_rule/0).
:- discontiguous(abstract_pattern/0).

neural_rule_1 :- feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.999, Extracted: 2025-04-10T08:02:36.777863, Activations: 1
neural_rule_10 :- feature_gradient(efficiency_index, _, high), feature_gradient(vibration, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.930, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_11 :- feature_gradient(pressure, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.954, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_12 :- feature_gradient(efficiency_index, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.941, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_13 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.891, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_14 :- correlated(temperature, vibration), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.939, Extracted: 2025-04-10T08:03:53.244338, Activations: 0
neural_rule_2 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.982, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_3 :- feature_gradient(efficiency_index, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.920, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_4 :- feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 1.000, Extracted: 2025-04-10T08:02:36.777863, Activations: 1
neural_rule_5 :- feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.935, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_6 :- feature_gradient(efficiency_index, _, high), feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.865, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_7 :- feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.999, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_8 :- feature_gradient(efficiency_index, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.900, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
neural_rule_9 :- feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.892, Extracted: 2025-04-10T08:02:36.777863, Activations: 0
