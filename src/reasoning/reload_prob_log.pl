%% src/reasoning/reload_prob_log.pl
:- module(reload_prob_log, [reload_prob_log/0]).

% Compliant with ProbLog 2.2
% Handles configuration reloading without dynamic predicates

% Main reload predicate
reload_prob_log :-
    % Load configuration
    consult('prob_rules.pl'),
    consult('integrate_prob_log.pl'),

    % Verify loading success
    check_configuration,

    % Run inference
    run_probabilistic_analysis.

% Configuration verification
check_configuration :-
    current_predicate(system_state/1),
    current_predicate(pattern_detected/1),
    current_predicate(generate_insight/1).

% Run analysis without state management
run_probabilistic_analysis :-
    findall(State-Risk, system_analysis(State, Risk), _),
    findall(Pattern, pattern_detected(Pattern), _),
    findall(Insight, generate_insight(Insight), _).

% Error handling
:- set_prolog_flag(unknown, fail).