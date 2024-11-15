% src/reasoning/rules.pl

% Rule: Degraded State
% Condition: Temperature > 80 and Vibration > 55
degraded_state(Temperature, Vibration) :-
    Temperature > 80,
    Vibration > 55.

% Rule: System Stress
% Condition: Pressure < 20
system_stress(Pressure) :-
    Pressure < 20.

% Rule: Critical State
% Condition: Efficiency Index < 0.6
critical_state(EfficiencyIndex) :-
    EfficiencyIndex < 0.6.

% Rule: Maintenance Required
% Condition: Operational Hours is a multiple of 1000
maintenance_needed(OperationalHours) :-
    0 is OperationalHours mod 1000.
