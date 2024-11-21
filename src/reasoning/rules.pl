%% src/reasoning/rules.pl - NEXUS-DT Symbolic Reasoning Rules

%% NEXUS-DT Symbolic Reasoning Rules
%% Compliant with ProbLog 2.2

% Import ProbLog integration
:- use_module(library(problog)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base State Rules with Probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% State rules with probabilities
0.9::degraded_state(Temperature, Vibration) :-
    Temperature > 80,
    Vibration > 55.

0.85::system_stress(Pressure) :-
    Pressure < 20.

0.8::critical_state(EfficiencyIndex) :-
    EfficiencyIndex < 0.6.

0.7::maintenance_needed(OperationalHours) :-
    0 is OperationalHours mod 1000.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature Threshold Definitions
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

% State transition probabilities
0.8::state_transition(From, To) :-
    member(From, [0,1,2]),
    member(To, [0,1,2]),
    From \= To.

0.7::compound_state_transition(From, Mid, To) :-
    state_transition(From, Mid),
    state_transition(Mid, To),
    From \= To.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gradient Analysis Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Feature gradients with probabilities
0.75::feature_gradient(temperature, Gradient, high) :-
    Gradient > 2.0.

0.75::feature_gradient(vibration, Gradient, high) :-
    Gradient > 1.5.

0.75::feature_gradient(pressure, Gradient, high) :-
    Gradient > 1.0.

0.75::feature_gradient(efficiency_index, Gradient, high) :-
    Gradient > 0.1.

% Rapid change detection
0.8::rapid_change(temperature, Old, New) :-
    abs(New - Old) > 10.

0.8::rapid_change(vibration, Old, New) :-
    abs(New - Old) > 5.

% Thermal patterns
0.85::rapid_temp_change(Old, New, Gradient) :-
    Gradient is abs(New - Old),
    Gradient > 2.0.

0.9::thermal_stress(Temp, Gradient) :-
    Temp > 75,
    rapid_temp_change(_, Temp, Gradient).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern Detection Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Critical patterns with probabilities
0.95::critical_pattern(Temp, Vib, Press, Eff) :-
    feature_threshold(temperature, Temp, high),
    feature_threshold(vibration, Vib, high),
    feature_threshold(pressure, Press, low),
    feature_threshold(efficiency_index, Eff, low).

% Sensor correlations
0.8::sensor_correlation(Temp, Vib, Press) :-
    Temp > 70,
    Vib > 45,
    Press < 25.

% Combined conditions
0.85::combined_condition(temperature, Temp, vibration, Vib) :-
    Temp > 75,
    Vib > 50.

0.8::combined_condition(pressure, Press, efficiency_index, Eff) :-
    Press < 25,
    Eff < 0.7.

% Multi-sensor patterns
0.9::multi_sensor_gradient(Temp_grad, Vib_grad, Press_grad) :-
    feature_gradient(temperature, Temp_grad, high),
    feature_gradient(vibration, Vib_grad, high),
    feature_gradient(pressure, Press_grad, high).

% State transition patterns
0.85::state_gradient_pattern(From, To, Gradient) :-
    state_transition(From, To),
    feature_gradient(temperature, Gradient, high).

% Efficiency patterns
0.8::efficiency_degradation(Eff, Grad) :-
    feature_threshold(efficiency_index, Eff, low),
    feature_gradient(efficiency_index, Grad, high).

% Cascade patterns
0.95::cascade_pattern(Temp, Vib, Press_grad, Time, Steps) :-
    Steps > 2,
    feature_gradient(temperature, Temp, high),
    feature_gradient(vibration, Vib, high),
    feature_gradient(pressure, Press_grad, high),
    maintenance_needed(Time),
    check_pressure(Press_grad).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pattern Matching Support Rules
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pattern matching with probabilities
0.8::pattern_match(Features, Thresholds) :-
    check_thresholds(Features, Thresholds).

check_thresholds([], []).
check_thresholds([Feature-Value|Features], [Threshold|Thresholds]) :-
    feature_threshold(Feature, Value, Threshold),
    check_thresholds(Features, Thresholds).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Support Predicates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

check_pressure(Value) :-
    Value < 30.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Queries and Evidence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Query declarations
query(critical_pattern(_, _, _, _)).
query(sensor_correlation(_, _, _)).
query(multi_sensor_gradient(_, _, _)).
query(cascade_pattern(_, _, _, _, _)).
query(state_gradient_pattern(_, _, _)).
query(efficiency_degradation(_, _)).

% Evidence declarations for sensor readings
evidence(sensor_value(temperature, T), Value) :-
    feature_threshold(temperature, T, high).

evidence(sensor_value(vibration, V), Value) :-
    feature_threshold(vibration, V, high).

evidence(sensor_value(pressure, P), Value) :-
    feature_threshold(pressure, P, low).

evidence(sensor_value(efficiency_index, E), Value) :-
    feature_threshold(efficiency_index, E, low).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Rule Integration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Space for dynamically learned rules
% Example format:
% 0.8::neural_rule_1(Temperature, Vibration) :-
%     Temperature > 80,
%     Vibration > 55.

% New Neural-Extracted Rules


% Neural-derived probabilistic rules
% Generated: 2024-11-21T07:11:03

0.9612529873847961::low_press.
0.9724326729774475::low_press.
0.9645239114761353::low_press.
0.9978137016296387::low_press.
0.9974256753921509::low_press.
0.9996644258499146::low_press.
0.9998403191566467::low_press.
0.9998188018798828::low_press.
0.9999175071716309::low_press.
0.9999356269836426::low_press.
0.9999334216117859::low_press.
0.9998655319213867::low_press.
0.999762237071991::low_press.
0.9997523427009583::low_press.
0.9997637271881104::low_press.
0.9997708201408386::low_press.
0.9997914433479309::low_press.
0.9998108744621277::low_press.
0.9998024702072144::low_press.
0.999817430973053::low_press.
0.9997999668121338::low_press.
0.9998037815093994::low_press.
0.9997850060462952::low_press.
0.999790370464325::low_press.
0.9997792840003967::low_press.
0.999748945236206::low_press.
0.9997562766075134::low_press.
0.9997212886810303::low_press.
0.9997352361679077::low_press.
0.9996881484985352::low_press.
0.9997333884239197::low_press.
0.9997403025627136::low_press.
0.9997449517250061::low_press.
0.999747097492218::low_press.
0.9997360706329346::low_press.
0.9997431635856628::low_press.
0.9997425675392151::low_press.
0.9997681379318237::low_press.
0.9997782707214355::low_press.
0.9997674822807312::low_press.
0.9997803568840027::low_press.
0.999806821346283::low_press.
0.9998164176940918::low_press.
0.9998013377189636::low_press.
0.9997977614402771::low_press.
0.99977046251297::low_press.
0.9997844099998474::low_press.
0.9997904300689697::low_press.
0.9997929930686951::low_press.
0.9997753500938416::low_press.
0.9997650980949402::low_press.
0.8876440525054932::low_press.
0.9324121475219727::low_press.
0.9983088374137878::low_press.
0.9994984865188599::low_press.
0.9998252987861633::low_press.
0.9999096989631653::low_press.
0.9998787641525269::low_press.
0.9999015927314758::low_press.
0.9999356269836426::low_press.
0.999923586845398::low_press.
0.9998589158058167::low_press.
0.9997677206993103::low_press.
0.9997829794883728::low_press.
0.9997562170028687::low_press.
0.9997608661651611::low_press.
0.9997746348381042::low_press.
0.9997488856315613::low_press.
0.9998076558113098::low_press.
0.999803364276886::low_press.
0.9997949004173279::low_press.
0.9997997879981995::low_press.
0.9997621774673462::low_press.
0.999780535697937::low_press.
0.9997686147689819::low_press.
0.9997727870941162::low_press.
0.9997473955154419::low_press.
0.9997121691703796::low_press.
0.9997377991676331::low_press.
0.9997379183769226::low_press.
0.9997571110725403::low_press.
0.9997590184211731::low_press.
0.9997437000274658::low_press.
0.9997363686561584::low_press.
0.999744713306427::low_press.
0.999752402305603::low_press.
0.9997630715370178::low_press.
0.9997576475143433::low_press.
0.999756932258606::low_press.
0.9997599720954895::low_press.
0.999734103679657::low_press.
0.9997518658638::low_press.
0.9997702240943909::low_press.
0.9998006224632263::low_press.
0.9998098015785217::low_press.
0.9998084902763367::low_press.
0.9997671842575073::low_press.
0.9997559189796448::low_press.
0.9997538924217224::low_press.
0.9997510313987732::low_press.
0.9997547268867493::low_press.
0.9974366426467896::low_press.
0.9620034694671631::low_press.
0.9807299971580505::low_press.
0.8739749193191528::low_press.
0.9995434880256653::low_press.
0.9994493722915649::low_press.
0.9998343586921692::low_press.
0.9998915791511536::low_press.
0.9998816847801208::low_press.
0.9999185800552368::low_press.
0.9999388456344604::low_press.
0.9999290704727173::low_press.
0.9998571872711182::low_press.
0.9997361302375793::low_press.
0.9997323155403137::low_press.
0.9997689127922058::low_press.
0.9997771978378296::low_press.
0.9997799396514893::low_press.
0.9997787475585938::low_press.
0.9998046159744263::low_press.
0.9998034238815308::low_press.
0.9997887015342712::low_press.
0.9997784495353699::low_press.
0.9997307062149048::low_press.
0.9997382164001465::low_press.
0.999733030796051::low_press.
0.9997036457061768::low_press.
0.9997687339782715::low_press.
0.9997650384902954::low_press.
0.9997462034225464::low_press.
0.9997596740722656::low_press.
0.9997503757476807::low_press.
0.9997199773788452::low_press.
0.9997407793998718::low_press.
0.9997466802597046::low_press.
0.9997409582138062::low_press.
0.9997468590736389::low_press.
0.9997584223747253::low_press.
0.9997762441635132::low_press.
0.999784529209137::low_press.
0.9997773170471191::low_press.
0.9997364282608032::low_press.
0.9997647404670715::low_press.
0.9997549057006836::low_press.
0.9997577667236328::low_press.
0.999794602394104::low_press.
0.9998316168785095::low_press.
0.9998043179512024::low_press.
0.9998332262039185::low_press.
0.9997652769088745::low_press.
0.9997366070747375::low_press.
0.9997425079345703::low_press.
0.9612529873847961::press_eff_correlation :- low_press, low_efficiency.
0.9724326729774475::press_eff_correlation :- low_press, low_efficiency.
0.9645239114761353::press_eff_correlation :- low_press, low_efficiency.
0.9978137016296387::press_eff_correlation :- low_press, low_efficiency.
0.9974256753921509::press_eff_correlation :- low_press, low_efficiency.
0.9996644258499146::press_eff_correlation :- low_press, low_efficiency.
0.9998403191566467::press_eff_correlation :- low_press, low_efficiency.
0.9998188018798828::press_eff_correlation :- low_press, low_efficiency.
0.9999175071716309::press_eff_correlation :- low_press, low_efficiency.
0.9999356269836426::press_eff_correlation :- low_press, low_efficiency.
0.9999334216117859::press_eff_correlation :- low_press, low_efficiency.
0.9998655319213867::press_eff_correlation :- low_press, low_efficiency.
0.999762237071991::press_eff_correlation :- low_press, low_efficiency.
0.9997523427009583::press_eff_correlation :- low_press, low_efficiency.
0.9997637271881104::press_eff_correlation :- low_press, low_efficiency.
0.9997708201408386::press_eff_correlation :- low_press, low_efficiency.
0.9997914433479309::press_eff_correlation :- low_press, low_efficiency.
0.9998108744621277::press_eff_correlation :- low_press, low_efficiency.
0.9998024702072144::press_eff_correlation :- low_press, low_efficiency.
0.999817430973053::press_eff_correlation :- low_press, low_efficiency.
0.9997999668121338::press_eff_correlation :- low_press, low_efficiency.
0.9998037815093994::press_eff_correlation :- low_press, low_efficiency.
0.9997850060462952::press_eff_correlation :- low_press, low_efficiency.
0.999790370464325::press_eff_correlation :- low_press, low_efficiency.
0.9997792840003967::press_eff_correlation :- low_press, low_efficiency.
0.999748945236206::press_eff_correlation :- low_press, low_efficiency.
0.9997562766075134::press_eff_correlation :- low_press, low_efficiency.
0.9997212886810303::press_eff_correlation :- low_press, low_efficiency.
0.9997352361679077::press_eff_correlation :- low_press, low_efficiency.
0.9996881484985352::press_eff_correlation :- low_press, low_efficiency.
0.9997333884239197::press_eff_correlation :- low_press, low_efficiency.
0.9997403025627136::press_eff_correlation :- low_press, low_efficiency.
0.9997449517250061::press_eff_correlation :- low_press, low_efficiency.
0.999747097492218::press_eff_correlation :- low_press, low_efficiency.
0.9997360706329346::press_eff_correlation :- low_press, low_efficiency.
0.9997431635856628::press_eff_correlation :- low_press, low_efficiency.
0.9997425675392151::press_eff_correlation :- low_press, low_efficiency.
0.9997681379318237::press_eff_correlation :- low_press, low_efficiency.
0.9997782707214355::press_eff_correlation :- low_press, low_efficiency.
0.9997674822807312::press_eff_correlation :- low_press, low_efficiency.
0.9997803568840027::press_eff_correlation :- low_press, low_efficiency.
0.999806821346283::press_eff_correlation :- low_press, low_efficiency.
0.9998164176940918::press_eff_correlation :- low_press, low_efficiency.
0.9998013377189636::press_eff_correlation :- low_press, low_efficiency.
0.9997977614402771::press_eff_correlation :- low_press, low_efficiency.
0.99977046251297::press_eff_correlation :- low_press, low_efficiency.
0.9997844099998474::press_eff_correlation :- low_press, low_efficiency.
0.9997904300689697::press_eff_correlation :- low_press, low_efficiency.
0.9997929930686951::press_eff_correlation :- low_press, low_efficiency.
0.9997753500938416::press_eff_correlation :- low_press, low_efficiency.
0.9997650980949402::press_eff_correlation :- low_press, low_efficiency.
0.8876440525054932::press_eff_correlation :- low_press, low_efficiency.
0.9324121475219727::press_eff_correlation :- low_press, low_efficiency.
0.9983088374137878::press_eff_correlation :- low_press, low_efficiency.
0.9994984865188599::press_eff_correlation :- low_press, low_efficiency.
0.9998252987861633::press_eff_correlation :- low_press, low_efficiency.
0.9999096989631653::press_eff_correlation :- low_press, low_efficiency.
0.9998787641525269::press_eff_correlation :- low_press, low_efficiency.
0.9999015927314758::press_eff_correlation :- low_press, low_efficiency.
0.9999356269836426::press_eff_correlation :- low_press, low_efficiency.
0.999923586845398::press_eff_correlation :- low_press, low_efficiency.
0.9998589158058167::press_eff_correlation :- low_press, low_efficiency.
0.9997677206993103::press_eff_correlation :- low_press, low_efficiency.
0.9997829794883728::press_eff_correlation :- low_press, low_efficiency.
0.9997562170028687::press_eff_correlation :- low_press, low_efficiency.
0.9997608661651611::press_eff_correlation :- low_press, low_efficiency.
0.9997746348381042::press_eff_correlation :- low_press, low_efficiency.
0.9997488856315613::press_eff_correlation :- low_press, low_efficiency.
0.9998076558113098::press_eff_correlation :- low_press, low_efficiency.
0.999803364276886::press_eff_correlation :- low_press, low_efficiency.
0.9997949004173279::press_eff_correlation :- low_press, low_efficiency.
0.9997997879981995::press_eff_correlation :- low_press, low_efficiency.
0.9997621774673462::press_eff_correlation :- low_press, low_efficiency.
0.999780535697937::press_eff_correlation :- low_press, low_efficiency.
0.9997686147689819::press_eff_correlation :- low_press, low_efficiency.
0.9997727870941162::press_eff_correlation :- low_press, low_efficiency.
0.9997473955154419::press_eff_correlation :- low_press, low_efficiency.
0.9997121691703796::press_eff_correlation :- low_press, low_efficiency.
0.9997377991676331::press_eff_correlation :- low_press, low_efficiency.
0.9997379183769226::press_eff_correlation :- low_press, low_efficiency.
0.9997571110725403::press_eff_correlation :- low_press, low_efficiency.
0.9997590184211731::press_eff_correlation :- low_press, low_efficiency.
0.9997437000274658::press_eff_correlation :- low_press, low_efficiency.
0.9997363686561584::press_eff_correlation :- low_press, low_efficiency.
0.999744713306427::press_eff_correlation :- low_press, low_efficiency.
0.999752402305603::press_eff_correlation :- low_press, low_efficiency.
0.9997630715370178::press_eff_correlation :- low_press, low_efficiency.
0.9997576475143433::press_eff_correlation :- low_press, low_efficiency.
0.999756932258606::press_eff_correlation :- low_press, low_efficiency.
0.9997599720954895::press_eff_correlation :- low_press, low_efficiency.
0.999734103679657::press_eff_correlation :- low_press, low_efficiency.
0.9997518658638::press_eff_correlation :- low_press, low_efficiency.
0.9997702240943909::press_eff_correlation :- low_press, low_efficiency.
0.9998006224632263::press_eff_correlation :- low_press, low_efficiency.
0.9998098015785217::press_eff_correlation :- low_press, low_efficiency.
0.9998084902763367::press_eff_correlation :- low_press, low_efficiency.
0.9997671842575073::press_eff_correlation :- low_press, low_efficiency.
0.9997559189796448::press_eff_correlation :- low_press, low_efficiency.
0.9997538924217224::press_eff_correlation :- low_press, low_efficiency.
0.9997510313987732::press_eff_correlation :- low_press, low_efficiency.
0.9997547268867493::press_eff_correlation :- low_press, low_efficiency.
0.9974366426467896::press_eff_correlation :- low_press, low_efficiency.
0.9620034694671631::press_eff_correlation :- low_press, low_efficiency.
0.9807299971580505::press_eff_correlation :- low_press, low_efficiency.
0.8739749193191528::press_eff_correlation :- low_press, low_efficiency.
0.9995434880256653::press_eff_correlation :- low_press, low_efficiency.
0.9994493722915649::press_eff_correlation :- low_press, low_efficiency.
0.9998343586921692::press_eff_correlation :- low_press, low_efficiency.
0.9998915791511536::press_eff_correlation :- low_press, low_efficiency.
0.9998816847801208::press_eff_correlation :- low_press, low_efficiency.
0.9999185800552368::press_eff_correlation :- low_press, low_efficiency.
0.9999388456344604::press_eff_correlation :- low_press, low_efficiency.
0.9999290704727173::press_eff_correlation :- low_press, low_efficiency.
0.9998571872711182::press_eff_correlation :- low_press, low_efficiency.
0.9997361302375793::press_eff_correlation :- low_press, low_efficiency.
0.9997323155403137::press_eff_correlation :- low_press, low_efficiency.
0.9997689127922058::press_eff_correlation :- low_press, low_efficiency.
0.9997771978378296::press_eff_correlation :- low_press, low_efficiency.
0.9997799396514893::press_eff_correlation :- low_press, low_efficiency.
0.9997787475585938::press_eff_correlation :- low_press, low_efficiency.
0.9998046159744263::press_eff_correlation :- low_press, low_efficiency.
0.9998034238815308::press_eff_correlation :- low_press, low_efficiency.
0.9997887015342712::press_eff_correlation :- low_press, low_efficiency.
0.9997784495353699::press_eff_correlation :- low_press, low_efficiency.
0.9997307062149048::press_eff_correlation :- low_press, low_efficiency.
0.9997382164001465::press_eff_correlation :- low_press, low_efficiency.
0.999733030796051::press_eff_correlation :- low_press, low_efficiency.
0.9997036457061768::press_eff_correlation :- low_press, low_efficiency.
0.9997687339782715::press_eff_correlation :- low_press, low_efficiency.
0.9997650384902954::press_eff_correlation :- low_press, low_efficiency.
0.9997462034225464::press_eff_correlation :- low_press, low_efficiency.
0.9997596740722656::press_eff_correlation :- low_press, low_efficiency.
0.9997503757476807::press_eff_correlation :- low_press, low_efficiency.
0.9997199773788452::press_eff_correlation :- low_press, low_efficiency.
0.9997407793998718::press_eff_correlation :- low_press, low_efficiency.
0.9997466802597046::press_eff_correlation :- low_press, low_efficiency.
0.9997409582138062::press_eff_correlation :- low_press, low_efficiency.
0.9997468590736389::press_eff_correlation :- low_press, low_efficiency.
0.9997584223747253::press_eff_correlation :- low_press, low_efficiency.
0.9997762441635132::press_eff_correlation :- low_press, low_efficiency.
0.999784529209137::press_eff_correlation :- low_press, low_efficiency.
0.9997773170471191::press_eff_correlation :- low_press, low_efficiency.
0.9997364282608032::press_eff_correlation :- low_press, low_efficiency.
0.9997647404670715::press_eff_correlation :- low_press, low_efficiency.
0.9997549057006836::press_eff_correlation :- low_press, low_efficiency.
0.9997577667236328::press_eff_correlation :- low_press, low_efficiency.
0.999794602394104::press_eff_correlation :- low_press, low_efficiency.
0.9998316168785095::press_eff_correlation :- low_press, low_efficiency.
0.9998043179512024::press_eff_correlation :- low_press, low_efficiency.
0.9998332262039185::press_eff_correlation :- low_press, low_efficiency.
0.9997652769088745::press_eff_correlation :- low_press, low_efficiency.
0.9997366070747375::press_eff_correlation :- low_press, low_efficiency.
0.9997425079345703::press_eff_correlation :- low_press, low_efficiency.

% New Neural-Extracted Rules

