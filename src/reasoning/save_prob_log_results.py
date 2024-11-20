# src/reasoning/save_prob_log_results.py - Executes ProbLog Queries and Saves Results to JSON

import sys
import json
from problog.program import PrologString
from problog import get_evaluatable


def run_prob_log_queries():
    prob_rules_path = 'src/reasoning/prob_rules.pl'

    try:
        with open(prob_rules_path, 'r') as file:
            prolog_code = file.read()
    except FileNotFoundError:
        print(f"Error: ProbLog rules file not found at {prob_rules_path}")
        sys.exit(1)

    # Initialize ProbLog model
    model = PrologString(prolog_code)
    query = get_evaluatable().create_from(model)

    # Evaluate queries
    result = query.evaluate()

    # Extract probabilities
    failure_risk_prob = result.get('failure_risk', 0.0)
    system_stress_prob = result.get('system_stress', 0.0)
    efficiency_drop_prob = result.get('efficiency_drop', 0.0)

    # Prepare results
    results = {
        'failure_risk': failure_risk_prob,
        'system_stress': system_stress_prob,
        'efficiency_drop': efficiency_drop_prob
    }

    # Save results to JSON file
    output_file = 'results/prob_log_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"ProbLog results saved to {output_file}")


if __name__ == "__main__":
    run_prob_log_queries()
