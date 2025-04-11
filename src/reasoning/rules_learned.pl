% Dynamically learned rules for ANSR-DT - Automatically managed
% Last updated: 2025-04-11T06:57:51.873132

:- discontiguous(neural_rule/0).
:- discontiguous(gradient_rule/0).
:- discontiguous(pattern_rule/0).
:- discontiguous(abstract_pattern/0).

neural_rule_1 :- feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.998, Extracted: 2025-04-10T09:23:24.741809, Activations: 2
neural_rule_10 :- feature_gradient(efficiency_index, _, high), feature_gradient(vibration, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.929, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_11 :- feature_gradient(pressure, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.953, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_12 :- feature_gradient(efficiency_index, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.939, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_13 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.891, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_14 :- correlated(temperature, vibration), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.938, Extracted: 2025-04-10T09:24:38.425432, Activations: 1
neural_rule_15 :- feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 1.000, Extracted: 2025-04-11T06:56:36.426576, Activations: 1
neural_rule_16 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.982, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_17 :- feature_gradient(efficiency_index, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.920, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_18 :- feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 1.000, Extracted: 2025-04-11T06:56:36.426576, Activations: 1
neural_rule_19 :- feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.936, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_2 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.981, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_20 :- feature_gradient(efficiency_index, _, high), feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.865, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_21 :- feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 1.000, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_22 :- feature_gradient(efficiency_index, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.900, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_23 :- feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.892, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_24 :- feature_gradient(efficiency_index, _, high), feature_gradient(vibration, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.930, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_25 :- feature_gradient(pressure, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.954, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_26 :- feature_gradient(efficiency_index, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.941, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_27 :- feature_gradient(efficiency_index, _, high), feature_gradient(pressure, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.892, Extracted: 2025-04-11T06:56:36.426576, Activations: 0
neural_rule_28 :- correlated(temperature, vibration), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.940, Extracted: 2025-04-11T06:57:51.872908, Activations: 0
neural_rule_3 :- feature_gradient(efficiency_index, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.919, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_4 :- feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.999, Extracted: 2025-04-10T09:23:24.741809, Activations: 2
neural_rule_5 :- feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.935, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_6 :- feature_gradient(efficiency_index, _, high), feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.864, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_7 :- feature_gradient(pressure, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.999, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_8 :- feature_gradient(efficiency_index, _, high), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.899, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
neural_rule_9 :- feature_gradient(temperature, _, high), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), feature_threshold(temperature, _, low), feature_threshold(vibration, _, low).  % Confidence: 0.891, Extracted: 2025-04-10T09:23:24.741809, Activations: 0
