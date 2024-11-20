#!/usr/bin/env python3
import os
import sys
import logging

from problog.program import SimpleProgram
from problog.logic import Term
from problog import get_evaluatable

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(filename='logs/prob_log_queries.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def run_prob_log_queries():
    prob_rules_path = 'prob_rules.pl'

    try:
        with open(prob_rules_path, 'r') as file:
            prolog_code = file.read()
        logging.info("ProbLog rules loaded successfully.")
    except FileNotFoundError:
        logging.error(f"ProbLog rules file not found at {prob_rules_path}")
        print(f"Error: ProbLog rules file not found at {prob_rules_path}")
        sys.exit(1)

    # Initialize ProbLog model
    try:
        # Use SimpleProgram instead of PrologString:
        model = SimpleProgram()
        for line in prolog_code.split('\n'):
            if line.strip() and not line.strip().startswith('%'):  # Ignore comments and empty lines
                model.add_clause(Term.from_string(line))

        # Define the query/1 predicate
        model.add_clause(Term.from_string("query(X) :- X."))
        query = get_evaluatable().create_from(model)
        logging.info("ProbLog model initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize ProbLog model: {e}")
        print(f"Error: Failed to initialize ProbLog model: {e}")
        sys.exit(1)

    # Evaluate queries
    try:
        result = query.evaluate()
        logging.info("ProbLog queries evaluated successfully.")
    except Exception as e:
        logging.error(f"Error during ProbLog query evaluation: {e}")
        print(f"Error: During ProbLog query evaluation: {e}")
        sys.exit(1)

    # Extract probabilities
    try:
        failure_risk_prob = result.get(Term('failure_risk'), 0.0)
        system_stress_prob = result.get(Term('system_stress'), 0.0)
        efficiency_drop_prob = result.get(Term('efficiency_drop'), 0.0)
        logging.info(f"ProbLog results: failure_risk={failure_risk_prob}, "
                     f"system_stress={system_stress_prob}, "
                     f"efficiency_drop={efficiency_drop_prob}")
    except Exception as e:
        logging.error(f"Error extracting probabilities: {e}")
        print(f"Error: Extracting probabilities: {e}")
        sys.exit(1)

    # Output probabilities in a Prolog-friendly format
    output = (f"failure_risk:{failure_risk_prob}\n"
              f"system_stress:{system_stress_prob}\n"
              f"efficiency_drop:{efficiency_drop_prob}\n")
    print(output)


if __name__ == "__main__":
    run_prob_log_queries()
