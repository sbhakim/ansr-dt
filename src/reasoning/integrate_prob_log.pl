% integrate_prob_log.pl - Integrates ProbLog Queries into Prolog

% Import necessary libraries
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(lists)).

% Define path to Python interpreter and ProbLog script
python_interpreter('python3').  % Adjust if using a different Python version
prob_log_script('src/reasoning/prob_query.py').

% Predicate to execute ProbLog queries and retrieve probabilities
run_prob_log_queries(FailureRisk, SystemStress, EfficiencyDrop) :-
    python_interpreter(Python),
    prob_log_script(Script),
    process_create(path(Python), [Script], [stdout(pipe(Out)), stderr(pipe(Err))]),
    read_string(Out, _, OutString),
    read_string(Err, _, ErrString),
    close(Out),
    close(Err),
    (   ErrString \= ""
    ->  format('Error from ProbLog: ~w~n', [ErrString]),
        Fail = 1
    ;   Fail = 0
    ),
    (   Fail = 0
    ->  split_string(OutString, "\n", "", Lines),
        maplist(split_pair, Lines, Pairs),
        member(failure_risk:FRStr, Pairs),
        member(system_stress:SSStr, Pairs),
        member(efficiency_drop:EDStr, Pairs),
        number_string(FailureRisk, FRStr),
        number_string(SystemStress, SSStr),
        number_string(EfficiencyDrop, EDStr)
    ;   FailureRisk = 0.0,
        SystemStress = 0.0,
        EfficiencyDrop = 0.0
    ).

% Helper predicate to split 'key:value' into Key-Value pairs
split_pair(Line, Key-Value) :-
    split_string(Line, ":", "", [KeyStr, ValueStr]),
    atom_string(Key, KeyStr),
    atom_number(ValueStr, Value),
    Value >= 0,  % Ensure Value is a number
    Value =< 1.0, % Ensure Value is a valid probability
    Value > 0.0,  % Optionally filter out zero probabilities
    Value >= 0.0,  % Allow zero probabilities if needed
    Value =< 1.0.

% Example predicate to report anomalies based on ProbLog probabilities
report_probabilistic_anomalies :-
    run_prob_log_queries(FailureRisk, SystemStress, EfficiencyDrop),
    format('Probability of Failure Risk: ~2f~n', [FailureRisk]),
    format('Probability of System Stress: ~2f~n', [SystemStress]),
    format('Probability of Efficiency Drop: ~2f~n', [EfficiencyDrop]),
    ( FailureRisk > 0.5 ->
        format('Anomaly Detected: High Failure Risk.~n')
    ;   true
    ),
    ( SystemStress > 0.6 ->
        format('Anomaly Detected: System Stress.~n')
    ;   true
    ),
    ( EfficiencyDrop > 0.5 ->
        format('Anomaly Detected: Efficiency Drop.~n')
    ;   true
    ).
