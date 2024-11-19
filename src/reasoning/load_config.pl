% load_config.pl - Loads Configuration Settings from config.yaml

:- use_module(library(yaml)).
:- use_module(library(lists)).

% Predicate to load configuration
load_config :-
    yaml_read_file('configs/config.yaml', Config),
    Config.get(knowledge_graph, KGConfig),
    KGConfig.get(prob_log, ProbLogConfig),
    ProbLogConfig.get(python_interpreter, Python),
    ProbLogConfig.get(prob_log_script, ProbLogScript),
    ProbLogConfig.get(prob_log_rules_file, ProbLogRules),
    % Assert dynamic predicates or store in Prolog terms
    retractall(prob_log_python_interpreter(_)),
    retractall(prob_log_script(_)),
    retractall(prob_log_rules_file(_)),
    assert(prob_log_python_interpreter(Python)),
    assert(prob_log_script(ProbLogScript)),
    assert(prob_log_rules_file(ProbLogRules)),
    format('Configuration loaded successfully.~n').

% Dynamic predicates to store configurations
:- dynamic prob_log_python_interpreter/1.
:- dynamic prob_log_script/1.
:- dynamic prob_log_rules_file/1.
