% src/reasoning/prob_rules.pl - Probabilistic Logic Programming Rules for NEXUS-DT

% Probabilistic Facts
0.8::high_temperature.
0.7::high_vibration.
0.9::low_pressure.
0.95::maintenance_needed.

% Probabilistic Rules
0.6::failure_risk :- high_temperature, high_vibration.
0.7::system_stress :- low_pressure.
0.8::efficiency_drop :- high_temperature, low_pressure.
0.5::efficiency_drop :- high_vibration, low_pressure.

% Additional Probabilistic Rules
0.4::overheating :- high_temperature, system_stress.
0.3::maintenance_required :- maintenance_needed, efficiency_drop.

% Queries
query(failure_risk).
query(system_stress).
query(efficiency_drop).
query(overheating).
query(maintenance_required).
