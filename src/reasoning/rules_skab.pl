%% src/reasoning/rules_skab.pl
%% Separate rule base for SKAB experiments.
%% This preserves the existing ANSR-DT symbolic interface while using
%% thresholds aligned to the mapped SKAB signals.

:- discontiguous(neural_rule/0).
:- discontiguous(gradient_rule/0).
:- discontiguous(pattern_rule/0).
:- discontiguous(abstract_pattern/0).

:- dynamic current_sensor_value/2.
:- dynamic sensor_change/2.
:- dynamic current_state/1.
:- dynamic previous_state/1.

temperature(TargetValue) :-
    current_sensor_value(temperature, CurrentValue),
    Tolerance is 0.5,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

vibration(TargetValue) :-
    current_sensor_value(vibration, CurrentValue),
    Tolerance is 0.01,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

pressure(TargetValue) :-
    current_sensor_value(pressure, CurrentValue),
    Tolerance is 0.1,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

efficiency_index(TargetValue) :-
    current_sensor_value(efficiency_index, CurrentValue),
    Tolerance is 0.2,
    LowerBound is TargetValue - Tolerance,
    UpperBound is TargetValue + Tolerance,
    CurrentValue >= LowerBound,
    CurrentValue =< UpperBound.

temperature_change(TargetChange) :-
    sensor_change(temperature, CurrentChange),
    Tolerance is 0.2,
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.

vibration_change(TargetChange) :-
    sensor_change(vibration, CurrentChange),
    Tolerance is 0.01,
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.

pressure_change(TargetChange) :-
    sensor_change(pressure, CurrentChange),
    Tolerance is 0.05,
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.

efficiency_change(TargetChange) :-
    sensor_change(efficiency_index, CurrentChange),
    Tolerance is 0.1,
    LowerBound is TargetChange - Tolerance,
    UpperBound is TargetChange + Tolerance,
    CurrentChange >= LowerBound,
    CurrentChange =< UpperBound.

state_transition(FromState, ToState) :-
    previous_state(PrevStateValue),
    current_state(CurrentStateValue),
    PrevStateValue == FromState,
    CurrentStateValue == ToState.

combined_high_temp_vib :-
    current_sensor_value(temperature, Temp),
    current_sensor_value(vibration, Vib),
    Temp < 88.0,
    Vib > 0.18.

combined_low_press_eff :-
    current_sensor_value(pressure, Press),
    current_sensor_value(efficiency_index, Eff),
    Press < 30.0,
    Eff < 0.8.

maintenance_needed(_) :- fail.
base_maintenance_needed(_) :- fail.
correlated(_, _) :- fail.
sequence_pattern(_) :- fail.
trend(_, increasing).
trend(_, decreasing).

degraded_state(Temperature, Vibration) :-
    feature_threshold(temperature, Temperature, low),
    feature_threshold(vibration, Vibration, high).

system_stress(Pressure) :-
    feature_threshold(pressure, Pressure, low).

% Treat large flow changes as stress even when the absolute level stays moderate.
thermal_stress(_, GradientValue) :-
    feature_gradient(pressure, GradientValue, high).

critical_state(EfficiencyIndex) :-
    feature_threshold(efficiency_index, EfficiencyIndex, low).

feature_threshold(temperature, Value, high) :- nonvar(Value), Value > 90.0.
feature_threshold(temperature, Value, low) :- nonvar(Value), Value < 72.0.

feature_threshold(vibration, Value, high) :- nonvar(Value), Value > 0.18.
feature_threshold(vibration, Value, low) :- nonvar(Value), Value < 0.04.

feature_threshold(pressure, Value, high) :- nonvar(Value), Value > 110.0.
feature_threshold(pressure, Value, low) :- nonvar(Value), Value < 30.0.

feature_threshold(efficiency_index, Value, low) :- nonvar(Value), Value < 0.8.
feature_threshold(efficiency_index, Value, medium) :- nonvar(Value), Value >= 0.8, Value < 1.2.

feature_gradient(temperature, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 0.5.
feature_gradient(vibration, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 0.02.
feature_gradient(pressure, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 2.0.
feature_gradient(efficiency_index, Gradient, high) :- nonvar(Gradient), abs(Gradient) > 0.25.

critical_pattern(Temp, Vib, Press, Eff) :-
    feature_threshold(temperature, Temp, low),
    feature_threshold(vibration, Vib, high),
    feature_threshold(pressure, Press, low),
    feature_threshold(efficiency_index, Eff, low).

sensor_correlation_alert(Temp, Vib, Press) :-
    nonvar(Temp), nonvar(Vib), nonvar(Press),
    Temp < 88.0,
    Vib > 0.18,
    Press < 30.0.

multi_sensor_gradient(TempGrad, VibGrad, PressGrad) :-
    feature_gradient(temperature, TempGrad, high),
    feature_gradient(vibration, VibGrad, high),
    feature_gradient(pressure, PressGrad, high).

state_gradient_pattern(From, To, TempGradient) :-
    state_transition(From, To),
    feature_gradient(temperature, TempGradient, high).

efficiency_degradation(Eff, Grad) :-
    feature_threshold(efficiency_index, Eff, low),
    feature_gradient(efficiency_index, Grad, high).
