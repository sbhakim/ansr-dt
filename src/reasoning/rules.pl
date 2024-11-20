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

% Predicate to report anomalies using ProbLogs probabilistic inferences
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%