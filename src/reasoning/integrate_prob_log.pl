% src/reasoning/integrate_prob_log.pl
% Integrates ProbLog Queries into Prolog

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import Necessary Libraries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- use_module(library(process)).    % For executing external processes
:- use_module(library(readutil)).   % For reading from streams
:- use_module(library(lists)).      % For list manipulation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define Configuration Paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Path to the Python interpreter. Adjust if using a different version or path.
python_interpreter('python3').

% Path to the ProbLog Python script that executes queries.
prob_log_script('prob_query.py').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main Predicate: run_prob_log_queries/3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% run_prob_log_queries(+FailureRisk, +SystemStress, +EfficiencyDrop)
% Executes the ProbLog queries via the Python script and retrieves probabilities.
%
% Parameters:
% - FailureRisk: Variable to store the probability of failure risk.
% - SystemStress: Variable to store the probability of system stress.
% - EfficiencyDrop: Variable to store the probability of efficiency drop.

run_prob_log_queries(FailureRisk, SystemStress, EfficiencyDrop) :-
    % Retrieve the Python interpreter and script paths from the facts.
    python_interpreter(Python),
    prob_log_script(Script),

    % Execute the Python script using the defined interpreter.
    % Capture both stdout and stderr.
    process_create(path(Python), [Script], [stdout(pipe(Out)), stderr(pipe(Err))]),

    % Read the output and error streams.
    read_string(Out, _, OutString),
    read_string(Err, _, ErrString),

    % Close the streams to free resources.
    close(Out),
    close(Err),

    % Check if there were any errors during script execution.
    (   ErrString \= ""
    ->  % If there are errors, log them and set Fail flag.
        format('Error from ProbLog: ~w~n', [ErrString]),
        Fail = 1
    ;   % If no errors, proceed normally.
        Fail = 0
    ),

    % If execution was successful, parse the output.
    (   Fail = 0
    ->  % Split the output string into individual lines.
        split_string(OutString, "\n", "", Lines),

        % Apply split_pair to each line to get Key-Value pairs.
        maplist(split_pair, Lines, Pairs),

        % Extract specific probabilities from the pairs.
        (   member(failure_risk:FRStr, Pairs),
            member(system_stress:SSStr, Pairs),
            member(efficiency_drop:EDStr, Pairs)
        ->  % Convert string representations to numerical values.
            number_string(FailureRisk, FRStr),
            number_string(SystemStress, SSStr),
            number_string(EfficiencyDrop, EDStr)
        ;   % If any expected key is missing, default to 0.0.
            format('Warning: Missing expected probability keys. Defaulting to 0.0.~n'),
            FailureRisk = 0.0,
            SystemStress = 0.0,
            EfficiencyDrop = 0.0
        )
    ;   % If execution failed, default all probabilities to 0.0.
        FailureRisk = 0.0,
        SystemStress = 0.0,
        EfficiencyDrop = 0.0
    ).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Predicate: split_pair/2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% split_pair(+Line, -Key-Value)
% Splits a string of the format 'key:value' into a Key-Value pair.
%
% Parameters:
% - Line: The input string to split.
% - Key-Value: The resulting Key-Value pair.

split_pair(Line, Key-Value) :-
    % Split the line at the colon ':'.
    split_string(Line, ":", "", [KeyStr, ValueStr]),

    % Convert the key string to an atom.
    atom_string(Key, KeyStr),

    % Attempt to convert the value string to a number.
    (   atom_number(ValueStr, Value)
    ->  % Ensure the value is within the valid probability range.
        (   Value >= 0.0,
            Value =< 1.0
        ->  true
        ;   % If out of range, default to 0.0 and log a warning.
            format('Warning: Value for ~w is out of range (0.0 - 1.0). Defaulting to 0.0.~n', [Key]),
            Value = 0.0
        )
    ;   % If conversion fails, default to 0.0 and log a warning.
        format('Warning: Invalid value format for ~w. Defaulting to 0.0.~n', [Key]),
        Value = 0.0
    ),

    % Form the Key-Value pair.
    Key = Value.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of integrate_prob_log.pl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
