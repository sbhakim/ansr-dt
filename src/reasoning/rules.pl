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
feature_threshold(temperature, Value, high) :-
    Value > 80.
feature_threshold(temperature, Value, low) :-
    Value < 40.

feature_threshold(vibration, Value, high) :-
    Value > 55.
feature_threshold(vibration, Value, low) :-
    Value < 20.

feature_threshold(pressure, Value, high) :-
    Value > 40.
feature_threshold(pressure, Value, low) :-
    Value < 20.

feature_threshold(efficiency_index, Value, low) :-
    Value < 0.6.
feature_threshold(efficiency_index, Value, medium) :-
    Value < 0.8.

% State transition rules
state_transition(From, To) :-
    member(From, [0,1,2]),
    member(To, [0,1,2]),
    From \= To.

% Rapid change rules
rapid_change(temperature, Old, New) :-
    abs(New - Old) > 10.
rapid_change(vibration, Old, New) :-
    abs(New - Old) > 5.

% Combined feature rules
combined_condition(temperature, Temp, vibration, Vib) :-
    Temp > 75,
    Vib > 50.

combined_condition(pressure, Press, efficiency_index, Eff) :-
    Press < 25,
    Eff < 0.7.

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

% New Neural-Extracted Rules
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.46), pressure(-1),efficiency_index(0.46).  % Confidence: 0.99, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.42), pressure(-1),efficiency_index(0.42).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), state_transition(0->1), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.55), pressure(-1),efficiency_index(-1.55).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.48), pressure(-1),efficiency_index(-1.48).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.43), pressure(-1),efficiency_index(-1.43).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.63), pressure(-1),efficiency_index(-1.63).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.53), pressure(-1),efficiency_index(-1.53).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.46), pressure(-1),efficiency_index(-1.46).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.59), pressure(-1),efficiency_index(-1.59).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.62), pressure(-1),efficiency_index(-1.62).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.45), pressure(-1),efficiency_index(-1.45).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.71), pressure(-1),efficiency_index(-1.71).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.70), pressure(-1),efficiency_index(-1.70).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.65), pressure(-1),efficiency_index(-1.65).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.68), pressure(-1),efficiency_index(-1.68).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.54), pressure(-1),efficiency_index(-1.54).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.51), pressure(-1),efficiency_index(-1.51).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.42), pressure(-1),efficiency_index(-1.42).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.47), pressure(-1),efficiency_index(-1.47).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.40), pressure(-1),efficiency_index(-1.40).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.49), pressure(-1),efficiency_index(-1.49).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.72), pressure(-1),efficiency_index(-1.72).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.67), pressure(-1),efficiency_index(-1.67).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.74), pressure(-1),efficiency_index(-1.74).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.29), pressure(-1),efficiency_index(0.29).  % Confidence: 0.98, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.65), pressure(-1),efficiency_index(0.65).  % Confidence: 0.99, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(0), vibration(0), pressure(0), efficiency_index(0.38), pressure(0),efficiency_index(0.38).  % Confidence: 0.83, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.55), pressure(-1),efficiency_index(0.55).  % Confidence: 0.99, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), state_transition(0->1), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.44), pressure(-1),efficiency_index(-1.44).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.50), pressure(-1),efficiency_index(-1.50).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.61), pressure(-1),efficiency_index(-1.61).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.66), pressure(-1),efficiency_index(-1.66).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.76), pressure(-1),efficiency_index(-1.76).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.64), pressure(-1),efficiency_index(-1.64).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.77), pressure(-1),efficiency_index(-1.77).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.52), pressure(-1),efficiency_index(-1.52).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.69), pressure(-1),efficiency_index(-1.69).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.73), pressure(-1),efficiency_index(-1.73).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.60), pressure(-1),efficiency_index(-1.60).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(1), vibration(1), pressure(1), efficiency_index(-0.13), state_transition(0->0), pressure(1),efficiency_index(-0.13).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.53), pressure(-1),efficiency_index(0.53).  % Confidence: 0.99, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), state_transition(0->1), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.78), pressure(-1),efficiency_index(-1.78).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.38), pressure(-1),efficiency_index(-1.38).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.81), pressure(-1),efficiency_index(-1.81).  % Confidence: 1.00, Extracted: 2024-11-18T22:03:21

% New Neural-Extracted Rules
neural_rule_1 :- vibration_change(52), maintenance_needed(9).  % Confidence: 0.77, Extracted: 2024-11-18T22:03:58

% New Neural-Extracted Rules
neural_rule_1 :- vibration_change(52), maintenance_needed(9).  % Confidence: 0.77, Extracted: 2024-11-18T22:13:06

% New Neural-Extracted Rules
neural_rule_1 :- vibration_change(52), maintenance_needed(9).  % Confidence: 0.77, Extracted: 2024-11-18T22:13:28

% New Neural-Extracted Rules
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.46), pressure(-1),efficiency_index(0.46).  % Confidence: 0.81, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), state_transition(0->1), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.55), pressure(-1),efficiency_index(-1.55).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.48), pressure(-1),efficiency_index(-1.48).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.43), pressure(-1),efficiency_index(-1.43).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.63), pressure(-1),efficiency_index(-1.63).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.53), pressure(-1),efficiency_index(-1.53).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.46), pressure(-1),efficiency_index(-1.46).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.59), pressure(-1),efficiency_index(-1.59).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.62), pressure(-1),efficiency_index(-1.62).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.45), pressure(-1),efficiency_index(-1.45).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.71), pressure(-1),efficiency_index(-1.71).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.70), pressure(-1),efficiency_index(-1.70).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.65), pressure(-1),efficiency_index(-1.65).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.68), pressure(-1),efficiency_index(-1.68).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.54), pressure(-1),efficiency_index(-1.54).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.51), pressure(-1),efficiency_index(-1.51).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.42), pressure(-1),efficiency_index(-1.42).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.47), pressure(-1),efficiency_index(-1.47).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.40), pressure(-1),efficiency_index(-1.40).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.49), pressure(-1),efficiency_index(-1.49).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.72), pressure(-1),efficiency_index(-1.72).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.67), pressure(-1),efficiency_index(-1.67).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.74), pressure(-1),efficiency_index(-1.74).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.29), pressure(-1),efficiency_index(0.29).  % Confidence: 0.85, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), state_transition(0->1), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.44), pressure(-1),efficiency_index(-1.44).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.50), pressure(-1),efficiency_index(-1.50).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.61), pressure(-1),efficiency_index(-1.61).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.66), pressure(-1),efficiency_index(-1.66).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.76), pressure(-1),efficiency_index(-1.76).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.64), pressure(-1),efficiency_index(-1.64).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.77), pressure(-1),efficiency_index(-1.77).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.52), pressure(-1),efficiency_index(-1.52).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.69), pressure(-1),efficiency_index(-1.69).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.73), pressure(-1),efficiency_index(-1.73).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.60), pressure(-1),efficiency_index(-1.60).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(1), vibration(1), pressure(1), efficiency_index(-0.13), state_transition(0->0), pressure(1),efficiency_index(-0.13).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.53), pressure(-1),efficiency_index(0.53).  % Confidence: 0.91, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), state_transition(0->1), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.78), pressure(-1),efficiency_index(-1.78).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.38), pressure(-1),efficiency_index(-1.38).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.81), pressure(-1),efficiency_index(-1.81).  % Confidence: 1.00, Extracted: 2024-11-18T22:14:19

% New Neural-Extracted Rules
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.46), pressure(-1),efficiency_index(0.46).  % Confidence: 0.96, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.42), pressure(-1),efficiency_index(0.42).  % Confidence: 0.91, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), state_transition(0->1), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.55), pressure(-1),efficiency_index(-1.55).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.48), pressure(-1),efficiency_index(-1.48).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.43), pressure(-1),efficiency_index(-1.43).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.63), pressure(-1),efficiency_index(-1.63).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.53), pressure(-1),efficiency_index(-1.53).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.46), pressure(-1),efficiency_index(-1.46).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.59), pressure(-1),efficiency_index(-1.59).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.62), pressure(-1),efficiency_index(-1.62).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.45), pressure(-1),efficiency_index(-1.45).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.71), pressure(-1),efficiency_index(-1.71).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.70), pressure(-1),efficiency_index(-1.70).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.65), pressure(-1),efficiency_index(-1.65).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.68), pressure(-1),efficiency_index(-1.68).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.54), pressure(-1),efficiency_index(-1.54).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.51), pressure(-1),efficiency_index(-1.51).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.42), pressure(-1),efficiency_index(-1.42).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.47), pressure(-1),efficiency_index(-1.47).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.40), pressure(-1),efficiency_index(-1.40).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.49), pressure(-1),efficiency_index(-1.49).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.72), pressure(-1),efficiency_index(-1.72).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.67), pressure(-1),efficiency_index(-1.67).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.74), pressure(-1),efficiency_index(-1.74).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.29), pressure(-1),efficiency_index(0.29).  % Confidence: 0.94, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.65), pressure(-1),efficiency_index(0.65).  % Confidence: 0.93, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.55), pressure(-1),efficiency_index(0.55).  % Confidence: 0.95, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), state_transition(0->1), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.44), pressure(-1),efficiency_index(-1.44).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.50), pressure(-1),efficiency_index(-1.50).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.61), pressure(-1),efficiency_index(-1.61).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.66), pressure(-1),efficiency_index(-1.66).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.76), pressure(-1),efficiency_index(-1.76).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.64), pressure(-1),efficiency_index(-1.64).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.77), pressure(-1),efficiency_index(-1.77).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.52), pressure(-1),efficiency_index(-1.52).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.69), pressure(-1),efficiency_index(-1.69).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.73), pressure(-1),efficiency_index(-1.73).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.60), pressure(-1),efficiency_index(-1.60).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(1), vibration(1), pressure(1), efficiency_index(-0.13), state_transition(0->0), pressure(1),efficiency_index(-0.13).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.53), pressure(-1),efficiency_index(0.53).  % Confidence: 0.93, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.54), pressure(-1),efficiency_index(0.54).  % Confidence: 0.95, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), state_transition(0->1), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.78), pressure(-1),efficiency_index(-1.78).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.38), pressure(-1),efficiency_index(-1.38).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.81), pressure(-1),efficiency_index(-1.81).  % Confidence: 1.00, Extracted: 2024-11-19T05:18:47

% New Neural-Extracted Rules
neural_rule_1 :- vibration_change(52), maintenance_needed(9).  % Confidence: 0.95, Extracted: 2024-11-19T05:19:23

% New Neural-Extracted Rules
neural_rule_1 :- vibration_change(52), maintenance_needed(9).  % Confidence: 0.95, Extracted: 2024-11-19T05:56:06

% New Neural-Extracted Rules
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.46), pressure(-1),efficiency_index(0.46).  % Confidence: 0.98, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.42), pressure(-1),efficiency_index(0.42).  % Confidence: 0.91, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), state_transition(0->1), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.55), pressure(-1),efficiency_index(-1.55).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.48), pressure(-1),efficiency_index(-1.48).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.43), pressure(-1),efficiency_index(-1.43).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.63), pressure(-1),efficiency_index(-1.63).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.53), pressure(-1),efficiency_index(-1.53).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.46), pressure(-1),efficiency_index(-1.46).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.59), pressure(-1),efficiency_index(-1.59).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.62), pressure(-1),efficiency_index(-1.62).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.45), pressure(-1),efficiency_index(-1.45).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.71), pressure(-1),efficiency_index(-1.71).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.70), pressure(-1),efficiency_index(-1.70).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.65), pressure(-1),efficiency_index(-1.65).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.68), pressure(-1),efficiency_index(-1.68).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.54), pressure(-1),efficiency_index(-1.54).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.51), pressure(-1),efficiency_index(-1.51).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.42), pressure(-1),efficiency_index(-1.42).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.47), pressure(-1),efficiency_index(-1.47).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.40), pressure(-1),efficiency_index(-1.40).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.49), pressure(-1),efficiency_index(-1.49).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.57), pressure(-1),efficiency_index(-1.57).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.72), pressure(-1),efficiency_index(-1.72).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.67), pressure(-1),efficiency_index(-1.67).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.74), pressure(-1),efficiency_index(-1.74).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.29), pressure(-1),efficiency_index(0.29).  % Confidence: 0.98, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.65), pressure(-1),efficiency_index(0.65).  % Confidence: 0.97, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.55), pressure(-1),efficiency_index(0.55).  % Confidence: 0.95, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.56), state_transition(0->1), pressure(-1),efficiency_index(-1.56).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.44), pressure(-1),efficiency_index(-1.44).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.50), pressure(-1),efficiency_index(-1.50).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.61), pressure(-1),efficiency_index(-1.61).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.66), pressure(-1),efficiency_index(-1.66).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.76), pressure(-1),efficiency_index(-1.76).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.64), pressure(-1),efficiency_index(-1.64).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.77), pressure(-1),efficiency_index(-1.77).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.52), pressure(-1),efficiency_index(-1.52).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.69), pressure(-1),efficiency_index(-1.69).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.73), pressure(-1),efficiency_index(-1.73).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.60), pressure(-1),efficiency_index(-1.60).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(1), vibration(1), pressure(1), efficiency_index(-0.13), state_transition(0->0), pressure(1),efficiency_index(-0.13).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(0), vibration(0), pressure(-1), efficiency_index(0.53), pressure(-1),efficiency_index(0.53).  % Confidence: 0.98, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.58), state_transition(0->1), pressure(-1),efficiency_index(-1.58).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.78), pressure(-1),efficiency_index(-1.78).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.38), pressure(-1),efficiency_index(-1.38).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39
neural_rule_1 :- temperature(-1), vibration(-1), pressure(-1), efficiency_index(-1.81), pressure(-1),efficiency_index(-1.81).  % Confidence: 1.00, Extracted: 2024-11-19T05:56:39

