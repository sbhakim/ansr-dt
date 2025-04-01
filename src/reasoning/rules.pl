%% src/reasoning/rules.pl - ANSR-DT Symbolic Reasoning Rules
%% Contains base rules, feature thresholds, state transitions, pattern detection,
%% definitions for dynamically generated predicates, and integration stubs.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Directives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Suppress warnings about discontiguous clauses for dynamically added rules
% These directives should match the rule names generated by Python (e.g., neural_rule_X)
% Adjust prefixes if Python generation changes. Add more if other prefixes are used.
:- discontiguous(neural_rule/0).
:- discontiguous(gradient_rule/0).
:- discontiguous(pattern_rule/0).
:- discontiguous(abstract_pattern/0). % If analyze_neural_patterns is used

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Integration and Configuration Files (Optional)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These are often handled better by managing the Prolog process from Python.
% Consulting files within Prolog can sometimes lead to path issues or unexpected reloading.
% Keeping them commented out unless specifically needed and tested.

% % Import integration predicates for ProbLog
% :- consult(integrate_prob_log).
%
% % Import rule management predicates (if applicable)
% % :- consult(manage_prob_rules).
%
% % Import rule reloading predicates
% :- consult(reload_prob_log).
%
% % Configuration via dynamic facts asserted by Python is generally more robust
% :- dynamic config/2.
% set_default_config :-
%     retractall(config(_,_)),
%     assertz(config(python_interpreter, 'python3')),
%     assertz(config(prob_log_script, 'prob_query.py')),
%     assertz(config(prob_log_rules_file, 'prob_rules.pl')).
% :- initialization(set_default_config, program).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamic Fact Predicates (Asserted/Retracted by Python)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These predicates hold the *current* state of the system for rule evaluation.
% Python code MUST assert these before querying rules and retract them afterwards.

:- dynamic current_sensor_value/2. % Format: current_sensor_value(sensor_name, numeric_value).
                                   % Example: current_sensor_value(temperature, 85.3).
:- dynamic sensor_change/2.      % Format: sensor_change(sensor_name, absolute_change_value).
                                   % Example: sensor_change(temperature, 11.2).
:- dynamic current_state/1.        % Format: current_state(integer_state). % 0, 1, or 2
                                   % Example: current_state(1).
:- dynamic previous_state/1.       % Format: previous_state(integer_state).
                                   % Example: previous_state(0).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definitions for Dynamically Generated Predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These predicates define the *meaning* of the terms used in the dynamically
% generated rules created by reasoning.py's extract_rules_from_neural_model.
% They check the currently asserted dynamic facts.

% --- Sensor Value Checks ---
temperature(TargetValue) :-
    current_sensor_value(temperature, CurrentValue),
    % Allow some tolerance for float comparison, adjust as needed
    Tolerance is 1.0,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

vibration(TargetValue) :-
    current_sensor_value(vibration, CurrentValue),
    Tolerance is 1.0,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

pressure(TargetValue) :-
    current_sensor_value(pressure, CurrentValue),
    Tolerance is 1.0,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

efficiency_index(TargetValue) :-
    current_sensor_value(efficiency_index, CurrentValue),
    Tolerance is 0.02, % Tighter tolerance for index values
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

% --- Sensor Change Checks ---
temperature_change(TargetChange) :-
    sensor_change(temperature, CurrentChange),
    Tolerance is 1.0,
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound, % Check absolute change
    CurrentChange =< UpperBound.

vibration_change(TargetChange) :-
    sensor_change(vibration, CurrentChange),
    Tolerance is 1.0,
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.

% Add pressure_change/1, efficiency_change/1 if generated by Python
pressure_change(TargetChange) :-
    sensor_change(pressure, CurrentChange),
    Tolerance is 0.5, % Adjust tolerance
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.

efficiency_change(TargetChange) :-
    sensor_change(efficiency_index, CurrentChange),
    Tolerance is 0.01, % Adjust tolerance
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.


% --- State Transition Check ---
state_transition(FromState, ToState) :-
    previous_state(PrevStateValue), % Check asserted previous state
    current_state(CurrentStateValue), % Check asserted current state
    PrevStateValue == FromState,    % Ensure the transition matches the rule
    CurrentStateValue == ToState.

% --- Combined Pattern Checks ---
combined_high_temp_vib :- % Arity 0, checks current values
    current_sensor_value(temperature, Temp),
    current_sensor_value(vibration, Vib),
    Temp > 75.0, % Use thresholds from Python generation logic
    Vib > 50.0.

combined_low_press_eff :- % Arity 0, checks current values
    current_sensor_value(pressure, Press),
    current_sensor_value(efficiency_index, Eff),
    Press < 25.0,
    Eff < 0.7.

% --- Maintenance Check ---
% This predicate name matches the one generated in reasoning.py
maintenance_needed(TargetOpHours) :-
    current_sensor_value(operational_hours, CurrentOpHours),
    % Check if current hours are close to the target value AND modulo condition holds
    Tolerance is 10.0, % Allow check within 10 hours of the 1000 mark
    LowerBound is TargetOpHours - Tolerance,
    UpperBound is TargetOpHours + Tolerance,
    CurrentOpHours >= LowerBound,
    CurrentOpHours =< UpperBound,
    0 is round(CurrentOpHours) mod 1000. % Check the actual modulo condition

% --- Add definitions for any *other* predicates generated by ---
% --- `extract_rules_from_neural_model` if necessary       ---


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base System State Rules (Using Threshold Predicates for Clarity)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define degraded state based on defined thresholds
degraded_state(Temperature, Vibration) :-
    feature_threshold(temperature, Temperature, high),
    feature_threshold(vibration, Vibration, high).

% Define system stress based on defined thresholds
system_stress(Pressure) :-
    feature_threshold(pressure, Pressure, low).

% Define critical state based on defined thresholds
critical_state(EfficiencyIndex) :-
    feature_threshold(efficiency_index, EfficiencyIndex, low).

% Define maintenance need based on operational hours (Modulo check)
% Note: This duplicates the dynamically generated `maintenance_needed/1` check slightly,
% but can be kept as a base definition. The Python assertion makes the dynamic one work.
base_maintenance_needed(OperationalHours) :-
    Val is round(OperationalHours) mod 1000,
    Val == 0.

% Detect thermal stress using defined thresholds and gradients
thermal_stress(Temp, GradientValue) :-
    feature_threshold(temperature, Temp, high),
    feature_gradient(temperature, GradientValue, high).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Threshold Definitions (Helper Predicates)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These define what 'high', 'low', 'medium' mean for different features.

% Temperature thresholds
feature_threshold(temperature, Value, high) :- nonvar(Value), Value > 80.
feature_threshold(temperature, Value, low) :- nonvar(Value), Value < 40.

% Vibration thresholds
feature_threshold(vibration, Value, high) :- nonvar(Value), Value > 55.
feature_threshold(vibration, Value, low) :- nonvar(Value), Value < 20.

% Pressure thresholds
feature_threshold(pressure, Value, high) :- nonvar(Value), Value > 40.
feature_threshold(pressure, Value, low) :- nonvar(Value), Value < 20.

% Efficiency thresholds
feature_threshold(efficiency_index, Value, low) :- nonvar(Value), Value < 0.6.
feature_threshold(efficiency_index, Value, medium) :- nonvar(Value), Value >= 0.6, Value < 0.8.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Definitions (Helper Predicates)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These define what a 'high' gradient means. Adjust thresholds as needed.

feature_gradient(temperature, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 2.0.
feature_gradient(vibration, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 1.5.
feature_gradient(pressure, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 1.0.
feature_gradient(efficiency_index, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 0.1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Advanced Pattern Detection Rules (Using Helper Predicates)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Detect critical patterns using defined thresholds
critical_pattern(Temp, Vib, Press, Eff) :-
    feature_threshold(temperature, Temp, high),
    feature_threshold(vibration, Vib, high),
    feature_threshold(pressure, Press, low),
    feature_threshold(efficiency_index, Eff, low).

% Detect specific sensor correlation pattern
sensor_correlation_alert(Temp, Vib, Press) :-
    nonvar(Temp), nonvar(Vib), nonvar(Press), % Ensure values are bound
    Temp > 70,
    Vib > 45,
    Press < 25.

% Detect multi-sensor high gradient patterns using defined gradients
multi_sensor_gradient(Temp_grad, Vib_grad, Press_grad) :-
    feature_gradient(temperature, Temp_grad, high),
    feature_gradient(vibration, Vib_grad, high),
    feature_gradient(pressure, Press_grad, high).

% Detect state transitions occurring with high temperature gradient
state_gradient_pattern(From, To, TempGradient) :-
    state_transition(From, To), % Uses the base state_transition/2 definition
    feature_gradient(temperature, TempGradient, high).

% Detect efficiency degradation
efficiency_degradation(Eff, Grad) :-
    feature_threshold(efficiency_index, Eff, low),
    NegGrad is -Grad, % Check for negative gradient (drop)
    feature_gradient(efficiency_index, NegGrad, high). % Check if the drop magnitude is high

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temporal Trend Predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define what increasing/decreasing trends mean for different features

% --- Temporal Trend Detection ---
trend(temperature, increasing) :-
    current_sensor_value(temperature, CurrentTemp),
    sensor_change(temperature, Change),
    Change > 1.0.

trend(temperature, decreasing) :-
    current_sensor_value(temperature, CurrentTemp),
    sensor_change(temperature, Change),
    Change < -1.0.

trend(vibration, increasing) :-
    current_sensor_value(vibration, CurrentVib),
    sensor_change(vibration, Change),
    Change > 0.5.

trend(vibration, decreasing) :-
    current_sensor_value(vibration, CurrentVib),
    sensor_change(vibration, Change),
    Change < -0.5.

trend(pressure, increasing) :-
    current_sensor_value(pressure, CurrentPressure),
    sensor_change(pressure, Change),
    Change > 0.5.

trend(pressure, decreasing) :-
    current_sensor_value(pressure, CurrentPressure),
    sensor_change(pressure, Change),
    Change < -0.5.

trend(efficiency_index, decreasing) :-
    current_sensor_value(efficiency_index, CurrentEff),
    sensor_change(efficiency_index, Change),
    Change < -0.05.

% --- Enhanced Feature Correlation Predicates ---
correlated(temperature, vibration) :-
    current_sensor_value(temperature, Temp),
    current_sensor_value(vibration, Vib),
    NormalizedTemp is Temp / 80.0,
    NormalizedVib is Vib / 60.0,
    Diff is abs(NormalizedTemp - NormalizedVib),
    Diff < 0.2.

correlated(temperature, pressure) :-
    current_sensor_value(temperature, Temp),
    current_sensor_value(pressure, Press),
    NormalizedTemp is Temp / 80.0,
    NormalizedPress is Press / 40.0,
    Diff is abs(NormalizedTemp - NormalizedPress),
    Diff < 0.2.

correlated(vibration, pressure) :-
    current_sensor_value(vibration, Vib),
    current_sensor_value(pressure, Press),
    NormalizedVib is Vib / 60.0,
    NormalizedPress is Press / 40.0,
    Diff is abs(NormalizedVib - NormalizedPress),
    Diff < 0.2.

% --- Pattern Sequence Predicates ---
sequence_pattern(increasing_temp_decreasing_press) :-
    trend(temperature, increasing),
    trend(pressure, decreasing).

sequence_pattern(all_increasing) :-
    trend(temperature, increasing),
    trend(vibration, increasing),
    trend(pressure, increasing).

sequence_pattern(all_decreasing) :-
    trend(temperature, decreasing),
    trend(vibration, decreasing),
    trend(pressure, decreasing).

sequence_pattern(efficiency_drop_with_temp_rise) :-
    trend(temperature, increasing),
    trend(efficiency_index, decreasing).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ProbLog Integration (Optional - keep stubs if used)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Predicate to report anomalies using ProbLog's probabilistic inferences
report_anomalies :-
    report_probabilistic_anomalies. % Assumes this is defined in integrate_prob_log.pl

% Predicate to reload ProbLog rules and re-run queries
reload_and_report :-
    reload_prob_log. % Assumes this is defined in reload_prob_log.pl

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of rules.pl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMIC RULES SECTION MARKER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rules below this marker are managed by the Python SymbolicReasoner.
% Manual edits here will likely be overwritten.







%% START DYNAMIC RULES %%
%% Automatically managed section - Do not edit manually below this line %%
neural_rule_1 :- correlated(efficiency_index, performance_score), correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, performance_score), correlated(operational_hours, system_state), correlated(pressure, efficiency_index), correlated(pressure, operational_hours), correlated(pressure, performance_score), correlated(pressure, system_state), correlated(system_state, performance_score), correlated(temperature, efficiency_index), correlated(temperature, operational_hours), correlated(temperature, performance_score), correlated(temperature, pressure), correlated(temperature, system_state), correlated(temperature, vibration), correlated(vibration, efficiency_index), correlated(vibration, operational_hours), correlated(vibration, performance_score), correlated(vibration, pressure), correlated(vibration, system_state), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low).  % Confidence: 0.987, Extracted: 2025-04-01T15:54:03.871996, Activations: 0
neural_rule_2 :- correlated(efficiency_index, performance_score), correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, performance_score), correlated(operational_hours, system_state), correlated(pressure, efficiency_index), correlated(pressure, operational_hours), correlated(pressure, performance_score), correlated(pressure, system_state), correlated(system_state, performance_score), correlated(temperature, efficiency_index), correlated(temperature, operational_hours), correlated(temperature, performance_score), correlated(temperature, pressure), correlated(temperature, system_state), correlated(temperature, vibration), correlated(vibration, efficiency_index), correlated(vibration, operational_hours), correlated(vibration, performance_score), correlated(vibration, pressure), correlated(vibration, system_state), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), trend(temperature, medium).  % Confidence: 1.000, Extracted: 2025-04-01T15:54:03.871996, Activations: 0
neural_rule_3 :- feature_threshold(pressure, _, low).  % Confidence: 1.000, Extracted: 2025-04-01T15:54:03.871996, Activations: 0
neural_rule_4 :- correlated(efficiency_index, performance_score), correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, performance_score), correlated(operational_hours, system_state), correlated(pressure, efficiency_index), correlated(pressure, operational_hours), correlated(pressure, performance_score), correlated(pressure, system_state), correlated(system_state, performance_score), correlated(temperature, efficiency_index), correlated(temperature, operational_hours), correlated(temperature, performance_score), correlated(temperature, pressure), correlated(temperature, system_state), correlated(temperature, vibration), correlated(vibration, efficiency_index), correlated(vibration, operational_hours), correlated(vibration, performance_score), correlated(vibration, pressure), correlated(vibration, system_state), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), trend(temperature, medium).  % Confidence: 1.000, Extracted: 2025-04-01T15:56:10.733620, Activations: 0
neural_rule_5 :- correlated(efficiency_index, performance_score), correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, performance_score), correlated(operational_hours, system_state), correlated(pressure, efficiency_index), correlated(pressure, operational_hours), correlated(pressure, performance_score), correlated(pressure, system_state), correlated(system_state, performance_score), correlated(temperature, efficiency_index), correlated(temperature, operational_hours), correlated(temperature, performance_score), correlated(temperature, pressure), correlated(temperature, system_state), correlated(temperature, vibration), correlated(vibration, efficiency_index), correlated(vibration, operational_hours), correlated(vibration, performance_score), correlated(vibration, pressure), correlated(vibration, system_state), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low).  % Confidence: 0.999, Extracted: 2025-04-01T15:56:10.733620, Activations: 0
neural_rule_6 :- feature_threshold(pressure, _, low).  % Confidence: 1.000, Extracted: 2025-04-01T15:56:10.733620, Activations: 0
neural_rule_7 :- correlated(efficiency_index, performance_score), correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, performance_score), correlated(operational_hours, system_state), correlated(pressure, efficiency_index), correlated(pressure, operational_hours), correlated(pressure, performance_score), correlated(pressure, system_state), correlated(system_state, performance_score), correlated(temperature, efficiency_index), correlated(temperature, operational_hours), correlated(temperature, performance_score), correlated(temperature, pressure), correlated(temperature, system_state), correlated(temperature, vibration), correlated(vibration, efficiency_index), correlated(vibration, operational_hours), correlated(vibration, performance_score), correlated(vibration, pressure), correlated(vibration, system_state), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low).  % Confidence: 0.867, Extracted: 2025-04-01T15:57:50.857220, Activations: 0
neural_rule_8 :- correlated(efficiency_index, performance_score), correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, performance_score), correlated(operational_hours, system_state), correlated(pressure, efficiency_index), correlated(pressure, operational_hours), correlated(pressure, performance_score), correlated(pressure, system_state), correlated(system_state, performance_score), correlated(temperature, efficiency_index), correlated(temperature, operational_hours), correlated(temperature, performance_score), correlated(temperature, pressure), correlated(temperature, system_state), correlated(temperature, vibration), correlated(vibration, efficiency_index), correlated(vibration, operational_hours), correlated(vibration, performance_score), correlated(vibration, pressure), correlated(vibration, system_state), feature_gradient(vibration, _, high), feature_threshold(efficiency_index, _, low), feature_threshold(pressure, _, low), trend(temperature, medium).  % Confidence: 0.999, Extracted: 2025-04-01T15:57:50.857220, Activations: 0
neural_rule_9 :- correlated(efficiency_index, system_state), correlated(operational_hours, efficiency_index), correlated(operational_hours, system_state), feature_gradient(vibration, _, high), trend(temperature, increasing).  % Confidence: 0.840, Extracted: 2025-04-01T15:58:29.284589, Activations: 0

%% END DYNAMIC RULES %%
