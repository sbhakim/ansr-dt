% src/reasoning/rules.pl

% Base Rules
degraded_state(Temperature, Vibration) :-
    Temperature > 80,
    Vibration > 55.

system_stress(Pressure) :-
    Pressure < 20.

critical_state(EfficiencyIndex) :-
    EfficiencyIndex < 0.6.

maintenance_needed(OperationalHours) :-
    0 is OperationalHours mod 1000.

% Neural Rule Templates
feature_threshold(temperature, Value, Threshold) :-
    Value > Threshold.

feature_threshold(vibration, Value, Threshold) :-
    Value > Threshold.

feature_threshold(pressure, Value, Threshold) :-
    Value < Threshold.

feature_threshold(efficiency_index, Value, Threshold) :-
    Value < Threshold.

% Pattern Templates
pattern_match(Features, Thresholds) :-
    check_thresholds(Features, Thresholds).

check_thresholds([], []).
check_thresholds([Feature-Value|Features], [Threshold|Thresholds]) :-
    feature_threshold(Feature, Value, Threshold),
    check_thresholds(Features, Thresholds).

% Space for Learned Rules
% neural_rule_1 :- ... (will be added dynamically)
% pattern_rule_1 :- ... (will be added dynamically)
% Neurosymbolic Learned Rules

% New Neural-Extracted Rules


% New Neural-Extracted Rules

% New Neural-Extracted Rules

% New Neural-Extracted Rules

% New Neural-Extracted Rules

% New Neural-Extracted Rules

% New Neural-Extracted Rules

% New Neural-Extracted Rules
neural_rule_1 :- pressure(-1), efficiency_index(0.46).  % Confidence: 0.85, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(0.42).  % Confidence: 0.94, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.55).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.48).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.43).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.63).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.53).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.46).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.59).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.62).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.45).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.71).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.70).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.65).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.68).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.54).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.51).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.42).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.47).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.40).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.49).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.72).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.67).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.74).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(0.29).  % Confidence: 0.95, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1).  % Confidence: 0.95, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(0.55).  % Confidence: 0.81, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.44).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.50).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.61).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.66).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.76).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.64).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.77).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.52).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.69).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.73).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.60).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(1), efficiency_index(-0.13).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(0.53).  % Confidence: 0.97, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.78).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.38).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30
neural_rule_1 :- pressure(-1), efficiency_index(-1.81).  % Confidence: 1.00, Extracted: 2024-11-18T21:38:30

% New Neural-Extracted Rules
neural_rule_1 :- pressure(-1), efficiency_index(0.46).  % Confidence: 0.90, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.55).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.48).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.43).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.63).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.53).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.46).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.59).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.62).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.45).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.71).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.70).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.65).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.68).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.54).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.51).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.42).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.47).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.40).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.49).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.72).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.67).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.74).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(0.29).  % Confidence: 0.96, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1).  % Confidence: 0.93, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.44).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.50).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.61).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.66).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.76).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.64).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.77).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.52).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.69).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.73).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.60).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(1), efficiency_index(-0.13).  % Confidence: 0.98, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(0.53).  % Confidence: 0.91, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.78).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.38).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40
neural_rule_1 :- pressure(-1), efficiency_index(-1.81).  % Confidence: 1.00, Extracted: 2024-11-18T21:47:40

