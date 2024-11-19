%% rules.pl - NEXUS-DT Symbolic Reasoning Rules
%% Contains base rules, feature thresholds, state transitions, and pattern detection

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base System State Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

degraded_state(Temperature, Vibration) :-
    Temperature > 80,
    Vibration > 55.

system_stress(Pressure) :-
    Pressure < 20.

critical_state(EfficiencyIndex) :-
    EfficiencyIndex < 0.6.

maintenance_needed(OperationalHours) :-
    0 is OperationalHours mod 1000.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Threshold Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Temperature thresholds
feature_threshold(temperature, Value, high) :-
    Value > 80.
feature_threshold(temperature, Value, low) :-
    Value < 40.

% Vibration thresholds
feature_threshold(vibration, Value, high) :-
    Value > 55.
feature_threshold(vibration, Value, low) :-
    Value < 20.

% Pressure thresholds
feature_threshold(pressure, Value, high) :-
    Value > 40.
feature_threshold(pressure, Value, low) :-
    Value < 20.

% Efficiency thresholds
feature_threshold(efficiency_index, Value, low) :-
    Value < 0.6.
feature_threshold(efficiency_index, Value, medium) :-
    Value < 0.8.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% State Transition Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

state_transition(From, To) :-
    member(From, [0,1,2]),
    member(To, [0,1,2]),
    From \= To.

compound_state_transition(From, Mid, To) :-
    state_transition(From, Mid),
    state_transition(Mid, To),
    From \= To.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Analysis Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Feature gradients
feature_gradient(temperature, Gradient, high) :-
    Gradient > 2.0.
feature_gradient(vibration, Gradient, high) :-
    Gradient > 1.5.
feature_gradient(pressure, Gradient, high) :-
    Gradient > 1.0.
feature_gradient(efficiency_index, Gradient, high) :-
    Gradient > 0.1.

% Rapid changes
rapid_change(temperature, Old, New) :-
    abs(New - Old) > 10.
rapid_change(vibration, Old, New) :-
    abs(New - Old) > 5.

% Thermal gradients
rapid_temp_change(Old, New, Gradient) :-
    Gradient is abs(New - Old),
    Gradient > 2.0.

thermal_stress(Temp, Gradient) :-
    Temp > 75,
    rapid_temp_change(_, Temp, Gradient).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern Detection Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Critical patterns
critical_pattern(Temp, Vib, Press, Eff) :-
    feature_threshold(temperature, Temp, high),
    feature_threshold(vibration, Vib, high),
    feature_threshold(pressure, Press, low),
    feature_threshold(efficiency_index, Eff, low).

% Sensor correlations
sensor_correlation(Temp, Vib, Press) :-
    Temp > 70,
    Vib > 45,
    Press < 25.

% Combined feature patterns
combined_condition(temperature, Temp, vibration, Vib) :-
    Temp > 75,
    Vib > 50.

combined_condition(pressure, Press, efficiency_index, Eff) :-
    Press < 25,
    Eff < 0.7.

% Multi-sensor gradient patterns
multi_sensor_gradient(Temp_grad, Vib_grad, Press_grad) :-
    feature_gradient(temperature, Temp_grad, high),
    feature_gradient(vibration, Vib_grad, high),
    feature_gradient(pressure, Press_grad, high).

% State transitions with gradients
state_gradient_pattern(From, To, Gradient) :-
    state_transition(From, To),
    feature_gradient(temperature, Gradient, high).

% Temporal correlations
temporal_correlation(Temp, Vib, Press, Time) :-
    feature_threshold(temperature, Temp, high),
    feature_threshold(vibration, Vib, high),
    maintenance_needed(Time).

% Efficiency degradation
efficiency_degradation(Eff, Grad) :-
    feature_threshold(efficiency_index, Eff, low),
    feature_gradient(efficiency_index, Grad, high).

% Cascade patterns
cascade_pattern(Temp, Vib, Press, Time, Steps) :-
    Steps > 2,
    feature_gradient(temperature, Temp, high),
    feature_gradient(vibration, Vib, high),
    feature_gradient(pressure, Press, high),
    maintenance_needed(Time),
    check_pressure(Press).  % Example additional usage

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern Matching Support Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pattern_match(Features, Thresholds) :-
    check_thresholds(Features, Thresholds).

check_thresholds([], []).
check_thresholds([Feature-Value|Features], [Threshold|Thresholds]) :-
    feature_threshold(Feature, Value, Threshold),
    check_thresholds(Features, Thresholds).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamic Rules Section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Space for dynamically learned rules
% neural_rule_1 :- ... (added during runtime)
% pattern_rule_1 :- ... (added during runtime)% Confidence: 0.97
gradient_rule_1 :- pressure_gradient(2.25), pressure(-1).
% Confidence: 1.00
gradient_rule_2 :- temperature_gradient(2.10), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_3 :- vibration_gradient(2.23), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_4 :- efficiency_index_gradient(2.33), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_5 :- temperature_gradient(2.10), temperature(-1).
% Confidence: 1.00
gradient_rule_6 :- vibration_gradient(2.23), vibration(-1).
% Confidence: 1.00
gradient_rule_7 :- efficiency_index_gradient(2.33), efficiency_index(-1).
% Confidence: 1.00
gradient_rule_8 :- temperature_gradient(2.58), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_9 :- vibration_gradient(2.23), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_10 :- efficiency_index_gradient(2.10), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_11 :- temperature_gradient(2.58), temperature(-1).
% Confidence: 1.00
gradient_rule_12 :- vibration_gradient(2.23), vibration(-1).
% Confidence: 1.00
gradient_rule_13 :- efficiency_index_gradient(2.10), efficiency_index(-1).
% Confidence: 1.00
gradient_rule_14 :- temperature_gradient(2.07), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_15 :- vibration_gradient(2.20), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_16 :- efficiency_index_gradient(2.34), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_17 :- temperature_gradient(2.07), temperature(-1).
% Confidence: 1.00
gradient_rule_18 :- vibration_gradient(2.20), vibration(-1).
% Confidence: 1.00
gradient_rule_19 :- efficiency_index_gradient(2.34), efficiency_index(-1).
% Confidence: 0.94
pattern_rule_1 :- pressure(-1), efficiency_index(0.46).
% Confidence: 0.97
pattern_rule_2 :- pressure(-1), efficiency_index(0.42).
% Confidence: 0.94
pattern_rule_3 :- pressure(-1), efficiency_index(0.46).
% Confidence: 1.00
pattern_rule_4 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 0.99
pattern_rule_5 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_6 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_7 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_8 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_9 :- pressure(-1), efficiency_index(-1.43).
% Confidence: 1.00
pattern_rule_10 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_11 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_12 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_13 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_14 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_15 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_16 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_17 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_18 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_19 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_20 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_21 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_22 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_23 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_24 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_25 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_26 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_27 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_28 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_29 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_30 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_31 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_32 :- pressure(-1), efficiency_index(-1.42).
% Confidence: 1.00
pattern_rule_33 :- pressure(-1), efficiency_index(-1.47).
% Confidence: 1.00
pattern_rule_34 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_35 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_36 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_37 :- pressure(-1), efficiency_index(-1.40).
% Confidence: 1.00
pattern_rule_38 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_39 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_40 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_41 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_42 :- pressure(-1), efficiency_index(-1.72).
% Confidence: 1.00
pattern_rule_43 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_44 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_45 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_46 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_47 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_48 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_49 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_50 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_51 :- pressure(-1), efficiency_index(-1.74).
% Confidence: 0.96
pattern_rule_52 :- pressure(-1), efficiency_index(0.29).
% Confidence: 0.97
pattern_rule_53 :- pressure(-1), efficiency_index(0.65).
% Confidence: 0.88
pattern_rule_54 :- pressure(-1), efficiency_index(0.55).
% Confidence: 1.00
pattern_rule_55 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_56 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_57 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_58 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_59 :- pressure(-1), efficiency_index(-1.47).
% Confidence: 1.00
pattern_rule_60 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_61 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_62 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_63 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_64 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_65 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_66 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_67 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_68 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_69 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_70 :- pressure(-1), efficiency_index(-1.76).
% Confidence: 1.00
pattern_rule_71 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_72 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_73 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_74 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_75 :- pressure(-1), efficiency_index(-1.77).
% Confidence: 1.00
pattern_rule_76 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_77 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_78 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_79 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_80 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_81 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_82 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_83 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_84 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_85 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_86 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_87 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_88 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_89 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_90 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_91 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_92 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_93 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_94 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_95 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_96 :- pressure(-1), efficiency_index(-1.72).
% Confidence: 1.00
pattern_rule_97 :- pressure(-1), efficiency_index(-1.74).
% Confidence: 1.00
pattern_rule_98 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_99 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_100 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_101 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_102 :- pressure(-1), efficiency_index(-1.60).
% Confidence: 1.00
pattern_rule_103 :- pressure(1), efficiency_index(-0.13).
% Confidence: 0.98
pattern_rule_104 :- pressure(-1), efficiency_index(0.65).
% Confidence: 0.98
pattern_rule_105 :- pressure(-1), efficiency_index(0.53).
% Confidence: 0.96
pattern_rule_106 :- pressure(-1), efficiency_index(0.54).
% Confidence: 1.00
pattern_rule_107 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_108 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_109 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_110 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_111 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_112 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_113 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_114 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_115 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_116 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_117 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_118 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_119 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_120 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_121 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_122 :- pressure(-1), efficiency_index(-1.78).
% Confidence: 1.00
pattern_rule_123 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_124 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_125 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_126 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_127 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_128 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_129 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_130 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_131 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_132 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_133 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_134 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_135 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_136 :- pressure(-1), efficiency_index(-1.38).
% Confidence: 1.00
pattern_rule_137 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_138 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_139 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_140 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_141 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_142 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_143 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_144 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_145 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_146 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_147 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_148 :- pressure(-1), efficiency_index(-1.81).
% Confidence: 1.00
pattern_rule_149 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_150 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_151 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_152 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_153 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_154 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 0.96
gradient_rule_1 :- pressure_gradient(2.25), pressure(-1).
% Confidence: 1.00
gradient_rule_2 :- temperature_gradient(2.10), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_3 :- vibration_gradient(2.23), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_4 :- efficiency_index_gradient(2.33), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_5 :- temperature_gradient(2.10), temperature(-1).
% Confidence: 1.00
gradient_rule_6 :- vibration_gradient(2.23), vibration(-1).
% Confidence: 1.00
gradient_rule_7 :- efficiency_index_gradient(2.33), efficiency_index(-1).
% Confidence: 1.00
gradient_rule_8 :- temperature_gradient(2.58), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_9 :- vibration_gradient(2.23), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_10 :- efficiency_index_gradient(2.10), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_11 :- temperature_gradient(2.58), temperature(-1).
% Confidence: 1.00
gradient_rule_12 :- vibration_gradient(2.23), vibration(-1).
% Confidence: 1.00
gradient_rule_13 :- efficiency_index_gradient(2.10), efficiency_index(-1).
% Confidence: 1.00
gradient_rule_14 :- temperature_gradient(2.07), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_15 :- vibration_gradient(2.20), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_16 :- efficiency_index_gradient(2.34), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_17 :- temperature_gradient(2.07), temperature(-1).
% Confidence: 1.00
gradient_rule_18 :- vibration_gradient(2.20), vibration(-1).
% Confidence: 1.00
gradient_rule_19 :- efficiency_index_gradient(2.34), efficiency_index(-1).
% Confidence: 0.97
pattern_rule_1 :- pressure(-1), efficiency_index(0.46).
% Confidence: 0.96
pattern_rule_2 :- pressure(-1), efficiency_index(0.42).
% Confidence: 0.95
pattern_rule_3 :- pressure(-1), efficiency_index(0.46).
% Confidence: 1.00
pattern_rule_4 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_5 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_6 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_7 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_8 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_9 :- pressure(-1), efficiency_index(-1.43).
% Confidence: 1.00
pattern_rule_10 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_11 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_12 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_13 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_14 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_15 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_16 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_17 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_18 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_19 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_20 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_21 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_22 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_23 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_24 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_25 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_26 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_27 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_28 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_29 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_30 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_31 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_32 :- pressure(-1), efficiency_index(-1.42).
% Confidence: 1.00
pattern_rule_33 :- pressure(-1), efficiency_index(-1.47).
% Confidence: 1.00
pattern_rule_34 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_35 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_36 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_37 :- pressure(-1), efficiency_index(-1.40).
% Confidence: 1.00
pattern_rule_38 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_39 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_40 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_41 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_42 :- pressure(-1), efficiency_index(-1.72).
% Confidence: 1.00
pattern_rule_43 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_44 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_45 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_46 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_47 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_48 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_49 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_50 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_51 :- pressure(-1), efficiency_index(-1.74).
% Confidence: 0.97
pattern_rule_52 :- pressure(-1), efficiency_index(0.29).
% Confidence: 0.82
pattern_rule_53 :- pressure(-1), efficiency_index(0.65).
% Confidence: 1.00
pattern_rule_54 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_55 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_56 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_57 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_58 :- pressure(-1), efficiency_index(-1.47).
% Confidence: 1.00
pattern_rule_59 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_60 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_61 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_62 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_63 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_64 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_65 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_66 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_67 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_68 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_69 :- pressure(-1), efficiency_index(-1.76).
% Confidence: 1.00
pattern_rule_70 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_71 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_72 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_73 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_74 :- pressure(-1), efficiency_index(-1.77).
% Confidence: 1.00
pattern_rule_75 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_76 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_77 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_78 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_79 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_80 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_81 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_82 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_83 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_84 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_85 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_86 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_87 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_88 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_89 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_90 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_91 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_92 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_93 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_94 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_95 :- pressure(-1), efficiency_index(-1.72).
% Confidence: 1.00
pattern_rule_96 :- pressure(-1), efficiency_index(-1.74).
% Confidence: 1.00
pattern_rule_97 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_98 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_99 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_100 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_101 :- pressure(-1), efficiency_index(-1.60).
% Confidence: 1.00
pattern_rule_102 :- pressure(1), efficiency_index(-0.13).
% Confidence: 0.97
pattern_rule_103 :- pressure(-1), efficiency_index(0.65).
% Confidence: 0.98
pattern_rule_104 :- pressure(-1), efficiency_index(0.53).
% Confidence: 1.00
pattern_rule_105 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_106 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_107 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_108 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_109 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_110 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_111 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_112 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_113 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_114 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_115 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_116 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_117 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_118 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_119 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_120 :- pressure(-1), efficiency_index(-1.78).
% Confidence: 1.00
pattern_rule_121 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_122 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_123 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_124 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_125 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_126 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_127 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_128 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_129 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_130 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_131 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_132 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_133 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_134 :- pressure(-1), efficiency_index(-1.38).
% Confidence: 1.00
pattern_rule_135 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_136 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_137 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_138 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_139 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_140 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_141 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_142 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_143 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_144 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_145 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_146 :- pressure(-1), efficiency_index(-1.81).
% Confidence: 1.00
pattern_rule_147 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_148 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_149 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_150 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_151 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_152 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 0.90
pattern_rule_153 :- pressure(0), efficiency_index(0.41).


% New Neural-Extracted Rules
neural_rule_1 :- vibration_change(52), maintenance_needed(9).  % Confidence: 0.95, Extracted: 2024-11-19T16:30:25

