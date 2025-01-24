%% src/reasoning/rules.pl - NEXUS-DT Symbolic Reasoning Rules
%% Enhanced with Probabilistic Logic Programming (ProbLog) Integration
%% Contains base rules, feature thresholds, state transitions, pattern detection,
%% and integration with ProbLog for probabilistic inferences.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Integration and Configuration Files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Import integration predicates for ProbLog
:- [integrate_prob_log].

% Import rule management predicates (if applicable)
% If you have not implemented manage_prob_rules.pl yet, comment out the following line
% :- [manage_prob_rules].

% Import rule reloading predicates
:- [reload_prob_log].

% Instead of YAML loading, use direct configuration
:- dynamic config/2.

% Set default configuration
set_default_config :-
    assert(config(python_interpreter, 'python3')),
    assert(config(prob_log_script, 'prob_query.py')),
    assert(config(prob_log_rules_file, 'prob_rules.pl')).

% Initialize configuration at startup
:- initialization(set_default_config).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base System State Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define degraded state based on temperature and vibration thresholds
degraded_state(Temperature, Vibration) :-
    Temperature > 80,
    Vibration > 55.

% Define system stress based on pressure threshold
system_stress(Pressure) :-
    Pressure < 20.

% Define critical state based on efficiency index
critical_state(EfficiencyIndex) :-
    EfficiencyIndex < 0.6.

% Define maintenance need based on operational hours
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

% Define possible state transitions
state_transition(From, To) :-
    member(From, [0,1,2]),
    member(To, [0,1,2]),
    From \= To.

% Define compound state transitions through intermediate states
compound_state_transition(From, Mid, To) :-
    state_transition(From, Mid),
    state_transition(Mid, To),
    From \= To.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Analysis Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Feature gradients indicating significant changes
feature_gradient(temperature, Gradient, high) :-
    Gradient > 2.0.
feature_gradient(vibration, Gradient, high) :-
    Gradient > 1.5.
feature_gradient(pressure, Gradient, high) :-
    Gradient > 1.0.
feature_gradient(efficiency_index, Gradient, high) :-
    Gradient > 0.1.

% Detect rapid changes in sensor readings
rapid_change(temperature, Old, New) :-
    abs(New - Old) > 10.
rapid_change(vibration, Old, New) :-
    abs(New - Old) > 5.

% Detect thermal gradients
rapid_temp_change(Old, New, Gradient) :-
    Gradient is abs(New - Old),
    Gradient > 2.0.

% Detect thermal stress based on temperature and gradient
thermal_stress(Temp, Gradient) :-
    Temp > 75,
    rapid_temp_change(_, Temp, Gradient).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern Detection Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Detect critical patterns based on multiple feature thresholds
critical_pattern(Temp, Vib, Press, Eff) :-
    feature_threshold(temperature, Temp, high),
    feature_threshold(vibration, Vib, high),
    feature_threshold(pressure, Press, low),
    feature_threshold(efficiency_index, Eff, low).

% Detect sensor correlations
sensor_correlation(Temp, Vib, Press) :-
    Temp > 70,
    Vib > 45,
    Press < 25.

% Detect combined feature patterns
combined_condition(temperature, Temp, vibration, Vib) :-
    Temp > 75,
    Vib > 50.

combined_condition(pressure, Press, efficiency_index, Eff) :-
    Press < 25,
    Eff < 0.7.

% Detect multi-sensor gradient patterns
multi_sensor_gradient(Temp_grad, Vib_grad, Press_grad) :-
    feature_gradient(temperature, Temp_grad, high),
    feature_gradient(vibration, Vib_grad, high),
    feature_gradient(pressure, Press_grad, high).

% Detect state transitions with gradients
state_gradient_pattern(From, To, Gradient) :-
    state_transition(From, To),
    feature_gradient(temperature, Gradient, high).

% Detect efficiency degradation
efficiency_degradation(Eff, Grad) :-
    feature_threshold(efficiency_index, Eff, low),
    feature_gradient(efficiency_index, Grad, high).

% Detect cascade patterns based on multiple conditions
cascade_pattern(Temp, Vib, Press_grad, Time, Steps) :-
    Steps > 2,
    feature_gradient(temperature, Temp, high),
    feature_gradient(vibration, Vib, high),
    feature_gradient(pressure, Press_grad, high),
    maintenance_needed(Time),
    check_pressure(Press_grad).  % Using Press_grad instead of Press to avoid singleton

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern Matching Support Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Match patterns based on feature thresholds
pattern_match(Features, Thresholds) :-
    check_thresholds(Features, Thresholds).

check_thresholds([], []).
check_thresholds([Feature-Value|Features], [Threshold|Thresholds]) :-
    feature_threshold(Feature, Value, Threshold),
    check_thresholds(Features, Thresholds).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Support Predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define check_pressure/1 predicate
check_pressure(Value) :-
    Value < 30.  % Define the threshold as per system requirements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamic Rules Section
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Space for dynamically learned rules
% neural_rule_1 :- ... (added during runtime)
% pattern_rule_1 :- ... (added during runtime) % Confidence: 0.97

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ProbLog Integration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Predicate to report anomalies using ProbLog's probabilistic inferences
report_anomalies :-
    report_probabilistic_anomalies.

% Predicate to reload ProbLog rules and re-run queries
reload_and_report :-
    reload_prob_log.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Additional Custom Rules (if any)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add any additional custom rules below as needed
% ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of rules.pl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Confidence: 1.00
gradient_rule_1 :- temperature_gradient(2.10), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_2 :- vibration_gradient(2.23), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_3 :- efficiency_index_gradient(2.33), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_4 :- temperature_gradient(2.10), temperature(-1).
% Confidence: 1.00
gradient_rule_5 :- vibration_gradient(2.23), vibration(-1).
% Confidence: 1.00
gradient_rule_6 :- efficiency_index_gradient(2.33), efficiency_index(-1).
% Confidence: 1.00
gradient_rule_7 :- temperature_gradient(2.58), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_8 :- vibration_gradient(2.23), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_9 :- efficiency_index_gradient(2.10), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_10 :- temperature_gradient(2.58), temperature(-1).
% Confidence: 1.00
gradient_rule_11 :- vibration_gradient(2.23), vibration(-1).
% Confidence: 1.00
gradient_rule_12 :- efficiency_index_gradient(2.10), efficiency_index(-1).
% Confidence: 1.00
gradient_rule_13 :- temperature_gradient(2.07), temperature(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_14 :- vibration_gradient(2.20), vibration(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_15 :- efficiency_index_gradient(2.34), efficiency_index(-1), state_transition(0->1).
% Confidence: 1.00
gradient_rule_16 :- temperature_gradient(2.07), temperature(-1).
% Confidence: 1.00
gradient_rule_17 :- vibration_gradient(2.20), vibration(-1).
% Confidence: 1.00
gradient_rule_18 :- efficiency_index_gradient(2.34), efficiency_index(-1).
% Confidence: 0.86
pattern_rule_1 :- pressure(-1), efficiency_index(0.46).
% Confidence: 0.84
pattern_rule_2 :- pressure(-1), efficiency_index(0.46).
% Confidence: 1.00
pattern_rule_3 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_4 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_5 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_6 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_7 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_8 :- pressure(-1), efficiency_index(-1.43).
% Confidence: 1.00
pattern_rule_9 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_10 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_11 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_12 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_13 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_14 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_15 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_16 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_17 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_18 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_19 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_20 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_21 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_22 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_23 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_24 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_25 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_26 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_27 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_28 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_29 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_30 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_31 :- pressure(-1), efficiency_index(-1.42).
% Confidence: 1.00
pattern_rule_32 :- pressure(-1), efficiency_index(-1.47).
% Confidence: 1.00
pattern_rule_33 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_34 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_35 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_36 :- pressure(-1), efficiency_index(-1.40).
% Confidence: 1.00
pattern_rule_37 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_38 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_39 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_40 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_41 :- pressure(-1), efficiency_index(-1.72).
% Confidence: 1.00
pattern_rule_42 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_43 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_44 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_45 :- pressure(-1), efficiency_index(-1.67).
% Confidence: 1.00
pattern_rule_46 :- pressure(-1), efficiency_index(-1.71).
% Confidence: 1.00
pattern_rule_47 :- pressure(-1), efficiency_index(-1.68).
% Confidence: 1.00
pattern_rule_48 :- pressure(-1), efficiency_index(-1.70).
% Confidence: 1.00
pattern_rule_49 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_50 :- pressure(-1), efficiency_index(-1.74).
% Confidence: 0.88
pattern_rule_51 :- pressure(-1), efficiency_index(0.29).
% Confidence: 0.86
pattern_rule_52 :- pressure(-1), efficiency_index(0.65).
% Confidence: 1.00
pattern_rule_53 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_54 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_55 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_56 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_57 :- pressure(-1), efficiency_index(-1.47).
% Confidence: 1.00
pattern_rule_58 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_59 :- pressure(-1), efficiency_index(-1.44).
% Confidence: 1.00
pattern_rule_60 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_61 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_62 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_63 :- pressure(-1), efficiency_index(-1.59).
% Confidence: 1.00
pattern_rule_64 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_65 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_66 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_67 :- pressure(-1), efficiency_index(-1.53).
% Confidence: 1.00
pattern_rule_68 :- pressure(-1), efficiency_index(-1.76).
% Confidence: 1.00
pattern_rule_69 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_70 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_71 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_72 :- pressure(-1), efficiency_index(-1.61).
% Confidence: 1.00
pattern_rule_73 :- pressure(-1), efficiency_index(-1.77).
% Confidence: 1.00
pattern_rule_74 :- pressure(-1), efficiency_index(-1.62).
% Confidence: 1.00
pattern_rule_75 :- pressure(-1), efficiency_index(-1.54).
% Confidence: 1.00
pattern_rule_76 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_77 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_78 :- pressure(-1), efficiency_index(-1.57).
% Confidence: 1.00
pattern_rule_79 :- pressure(-1), efficiency_index(-1.58).
% Confidence: 1.00
pattern_rule_80 :- pressure(-1), efficiency_index(-1.51).
% Confidence: 1.00
pattern_rule_81 :- pressure(-1), efficiency_index(-1.49).
% Confidence: 1.00
pattern_rule_82 :- pressure(-1), efficiency_index(-1.55).
% Confidence: 1.00
pattern_rule_83 :- pressure(-1), efficiency_index(-1.46).
% Confidence: 1.00
pattern_rule_84 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_85 :- pressure(-1), efficiency_index(-1.50).
% Confidence: 1.00
pattern_rule_86 :- pressure(-1), efficiency_index(-1.56).
% Confidence: 1.00
pattern_rule_87 :- pressure(-1), efficiency_index(-1.45).
% Confidence: 1.00
pattern_rule_88 :- pressure(-1), efficiency_index(-1.48).
% Confidence: 1.00
pattern_rule_89 :- pressure(-1), efficiency_index(-1.64).
% Confidence: 1.00
pattern_rule_90 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_91 :- pressure(-1), efficiency_index(-1.52).
% Confidence: 1.00
pattern_rule_92 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_93 :- pressure(-1), efficiency_index(-1.69).
% Confidence: 1.00
pattern_rule_94 :- pressure(-1), efficiency_index(-1.72).
% Confidence: 1.00
pattern_rule_95 :- pressure(-1), efficiency_index(-1.74).
% Confidence: 1.00
pattern_rule_96 :- pressure(-1), efficiency_index(-1.66).
% Confidence: 1.00
pattern_rule_97 :- pressure(-1), efficiency_index(-1.73).
% Confidence: 1.00
pattern_rule_98 :- pressure(-1), efficiency_index(-1.63).
% Confidence: 1.00
pattern_rule_99 :- pressure(-1), efficiency_index(-1.65).
% Confidence: 1.00
pattern_rule_100 :- pressure(-1), efficiency_index(-1.60).
% Confidence: 0.99
pattern_rule_101 :- pressure(1), efficiency_index(-0.13).
% Confidence: 0.91
pattern_rule_102 :- pressure(-1), efficiency_index(0.65).
% Confidence: 0.85
pattern_rule_103 :- pressure(-1), efficiency_index(0.53).
% Confidence: 0.83
pattern_rule_104 :- pressure(-1), efficiency_index(0.54).
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
