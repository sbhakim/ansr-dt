# src/reasoning/prob_query.py - Executes ProbLog Queries and Outputs Probabilities

import os
import sys
import logging
from problog.program import PrologString
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
        model = PrologString(prolog_code)
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
        failure_risk_prob = result.get('failure_risk', 0.0)
        system_stress_prob = result.get('system_stress', 0.0)
        efficiency_drop_prob = result.get('efficiency_drop', 0.0)
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
