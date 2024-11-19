# src/reasoning/prob_query.py


import sys
import logging
from problog.program import PrologString
from problog import get_evaluatable

# Configure logging
logging.basicConfig(filename='logs/prob_log_queries.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def run_prob_log_queries():
    prob_rules_path = 'src/reasoning/prob_rules.pl'

    try:
        with open(prob_rules_path, 'r') as file:
            prolog_code = file.read()
        logging.info("ProbLog rules loaded successfully.")
    except FileNotFoundError:
        logging.error(f"ProbLog rules file not found at {prob_rules_path}")
        print(f"Error: ProbLog rules file not found at {prob_rules_path}")
        sys.exit(1)

    # Initialize ProbLog model
    model = PrologString(prolog_code)
    query = get_evaluatable().create_from(model)

    try:
        # Evaluate queries
        result = query.evaluate()
        logging.info("ProbLog queries evaluated successfully.")
    except Exception as e:
        logging.error(f"Error during ProbLog query evaluation: {str(e)}")
        print(f"Error during ProbLog query evaluation: {str(e)}")
        sys.exit(1)

    # Extract probabilities
    failure_risk_prob = result.get('failure_risk', 0.0)
    system_stress_prob = result.get('system_stress', 0.0)
    efficiency_drop_prob = result.get('efficiency_drop', 0.0)

    # Output probabilities in a Prolog-friendly format
    output = f"failure_risk:{failure_risk_prob}\nsystem_stress:{system_stress_prob}\nefficiency_drop:{efficiency_drop_prob}\n"
    print(output)
    logging.info(
        f"ProbLog results: failure_risk={failure_risk_prob}, system_stress={system_stress_prob}, efficiency_drop={efficiency_drop_prob}")


if __name__ == "__main__":
    run_prob_log_queries()
