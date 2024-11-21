%% src/reasoning/integrate_prob_log.pl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEXUS-DT ProbLog Integration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NEXUS-DT ProbLog Integration
%% Compliant with ProbLog 2.2

% Import required ProbLog libraries
:- use_module(library(problog)).

% Load sensor monitoring rules
:- [prob_rules].

% Constants for thresholds
threshold(temperature, high, 80.0).
threshold(temperature, low, 40.0).
threshold(vibration, high, 55.0).
threshold(vibration, low, 20.0).
threshold(pressure, high, 40.0).
threshold(pressure, low, 20.0).
threshold(efficiency_index, low, 0.6).

% Probabilistic sensor analysis
0.9::sensor_above_threshold(Sensor, Type) :-
    threshold(Sensor, Type, Thresh),
    sensor_value(Sensor, Value),
    Value > Thresh.

0.9::sensor_below_threshold(Sensor, Type) :-
    threshold(Sensor, Type, Thresh),
    sensor_value(Sensor, Value),
    Value < Thresh.

% System state determination
system_state(critical) :-
    sensor_above_threshold(temperature, high),
    sensor_above_threshold(vibration, high),
    sensor_below_threshold(pressure, low),
    sensor_below_threshold(efficiency_index, low).

system_state(degraded) :-
    sensor_above_threshold(temperature, high),
    sensor_above_threshold(vibration, high),
    \+ system_state(critical).

system_state(normal) :-
    \+ system_state(critical),
    \+ system_state(degraded).

% Probabilistic correlation detection
0.8::sensor_correlation(Sensor1, Sensor2) :-
    sensor_above_threshold(Sensor1, high),
    sensor_above_threshold(Sensor2, high).

% Pattern analysis
0.85::pattern_detected(temp_vib) :-
    sensor_correlation(temperature, vibration).

0.75::pattern_detected(press_eff) :-
    sensor_below_threshold(pressure, low),
    sensor_below_threshold(efficiency_index, low).

% Risk assessment
0.9::high_risk :-
    system_state(critical).

0.7::medium_risk :-
    system_state(degraded).

0.2::low_risk :-
    system_state(normal).

% Query declarations
query(system_state(State)).
query(pattern_detected(Pattern)).
query(high_risk).
query(medium_risk).
query(low_risk).

% Evidence declarations
evidence(sensor_above_threshold(temperature, high), true) :-
    sensor_value(temperature, Value),
    threshold(temperature, high, Thresh),
    Value > Thresh.

evidence(sensor_above_threshold(vibration, high), true) :-
    sensor_value(vibration, Value),
    threshold(vibration, high, Thresh),
    Value > Thresh.

evidence(sensor_below_threshold(pressure, low), true) :-
    sensor_value(pressure, Value),
    threshold(pressure, low, Thresh),
    Value < Thresh.

% Combined analysis predicates
system_analysis(State, Risk) :-
    system_state(State),
    (State = critical -> high_risk;
     State = degraded -> medium_risk;
     low_risk).

% Probabilistic insights
0.8::requires_attention :-
    system_state(critical);
    (system_state(degraded), pattern_detected(_)).

0.7::requires_monitoring :-
    system_state(degraded);
    pattern_detected(_).

% Anomaly detection
anomaly_detected :-
    requires_attention;
    (requires_monitoring, sensor_correlation(_, _)).

% Insight generation
generate_insight(critical_alert) :-
    system_state(critical),
    requires_attention.

generate_insight(degradation_warning) :-
    system_state(degraded),
    requires_monitoring.

generate_insight(correlation_alert) :-
    sensor_correlation(Sensor1, Sensor2),
    requires_monitoring.

% Query interface
analyze_system :-
    findall(State-Risk, system_analysis(State, Risk), Analysis),
    findall(Insight, generate_insight(Insight), Insights).

% Exported predicates for Python interface
:- export(system_state/1).
:- export(pattern_detected/1).
:- export(generate_insight/1).
:- export(analyze_system/0).
:- export(anomaly_detected/0).