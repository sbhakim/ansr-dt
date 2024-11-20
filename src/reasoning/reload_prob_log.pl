% src/reasoning/reload_prob_log.pl

reload_prob_log :-
    retractall(prob_log_python_interpreter(_)),
    retractall(prob_log_script(_)),
    retractall(prob_log_rules_file(_)),
    load_config,
    format('ProbLog configuration reloaded.~n'),
    format('Re-inferencing with updated ProbLog rules.~n'),
    report_probabilistic_anomalies.
