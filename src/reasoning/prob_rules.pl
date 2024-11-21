%% src/reasoning/prob_rules.pl

% NEXUS-DT Probabilistic Rules
% Compliant with ProbLog 2.2

% Probabilistic facts with :: notation
0.8::high_temp.
0.7::high_vib.
0.9::low_press.
0.85::normal_eff.

% State definitions using probabilistic reasoning
system_state(normal) :-
    \+ high_temp,
    \+ high_vib,
    \+ low_press,
    normal_eff.

system_state(degraded) :-
    high_temp,
    high_vib,
    \+ low_press.

system_state(critical) :-
    high_temp,
    high_vib,
    low_press,
    \+ normal_eff.

% Probabilistic rules for failure prediction
0.9::failure_risk :- high_temp, high_vib.
0.8::system_stress :- low_press.

% Efficiency rules with probabilities
0.7::efficiency_drop :-
    high_temp,
    low_press.

0.6::efficiency_drop :-
    high_vib,
    low_press.

% Pattern detection with probabilistic reasoning
abnormal_pattern :-
    high_temp,
    high_vib,
    low_press,
    \+ normal_eff.

% Safety critical conditions
safety_critical :-
    high_temp,
    high_vib,
    low_press,
    \+ normal_eff.

% Performance monitoring
performance_degraded :-
    efficiency_drop,
    (system_stress; failure_risk).

% Thresholds as deterministic facts
threshold(temperature, high, 80.0).
threshold(vibration, high, 55.0).
threshold(pressure, low, 20.0).
threshold(efficiency, low, 0.6).

% Evidence predicates
evidence(temp_high) :-
    threshold(temperature, high, Thresh),
    query(failure_risk).

evidence(vib_high) :-
    threshold(vibration, high, Thresh),
    query(system_stress).

evidence(press_low) :-
    threshold(pressure, low, Thresh),
    query(efficiency_drop).

evidence(eff_normal) :-
    threshold(efficiency, low, Thresh).

% Queries
query(failure_risk).
query(system_stress).
query(efficiency_drop).
query(system_state(State)).
query(abnormal_pattern).
query(safety_critical).
query(performance_degraded).

% Neural-derived rules can be added below using :: notation
% Example:
% 0.75::neural_rule_1 :- high_temp, high_vib, low_press.