# src/reasoning/prob_log_interface.py

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from problog.program import PrologString, SimpleProgram
from problog.logic import Term
from problog import get_evaluatable


class ProbLogInterface:
    def __init__(self, rules_path: str = 'src/reasoning/prob_rules.pl', logger: Optional[logging.Logger] = None):
        """
        Initialize ProbLog interface.

        Args:
            rules_path (str): Path to ProbLog rules file
            logger (logging.Logger): Optional logger instance
        """
        self.rules_path = rules_path
        self.logger = logger or self._setup_logger()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._validate_paths()

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            os.makedirs('logs', exist_ok=True)
            handler = logging.FileHandler('logs/prob_log_queries.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _validate_paths(self) -> None:
        """Validate existence of required files and directories."""
        if not os.path.exists(self.rules_path):
            error_msg = f"ProbLog rules file not found at {self.rules_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _load_rules(self) -> SimpleProgram:
        """Load ProbLog rules from file."""
        try:
            # Add explicit library import
            model = SimpleProgram()
            model.add_clause(Term.from_string(":- use_module(library(problog))."))

            with open(self.rules_path, 'r') as file:
                prolog_code = file.read()

            for line in prolog_code.split('\n'):
                if line.strip() and not line.strip().startswith('%'):
                    model.add_clause(Term.from_string(line))

            return model
        except Exception as e:
            self.logger.error(f"Failed to load ProbLog rules: {e}")
            raise

    def _evaluate_queries(self, model: SimpleProgram) -> Dict[str, float]:
        """
        Evaluate ProbLog queries from the model.

        Args:
            model (SimpleProgram): Initialized ProbLog model

        Returns:
            Dict[str, float]: Query results with probabilities
        """
        try:
            query = get_evaluatable().create_from(model)
            result = query.evaluate()

            # Extract probabilities for standard queries
            probabilities = {
                'system_state_normal': result.get(Term('system_state(normal)'), 0.0),
                'system_state_degraded': result.get(Term('system_state(degraded)'), 0.0),
                'system_state_critical': result.get(Term('system_state(critical)'), 0.0),
                'failure_risk': result.get(Term('failure_risk'), 0.0),
                'system_stress': result.get(Term('system_stress'), 0.0),
                'efficiency_drop': result.get(Term('efficiency_drop'), 0.0),
                'abnormal_pattern': result.get(Term('abnormal_pattern'), 0.0),
                'safety_critical': result.get(Term('safety_critical'), 0.0),
                'performance_degraded': result.get(Term('performance_degraded'), 0.0)
            }

            self.logger.info("ProbLog queries evaluated successfully")
            return probabilities

        except Exception as e:
            self.logger.error(f"Query evaluation failed: {e}")
            raise

    def run_single_query(self, timeout: int = 30) -> Dict[str, float]:
        """
        Execute a single ProbLog query with timeout.

        Args:
            timeout (int): Query timeout in seconds

        Returns:
            Dict[str, float]: Query results
        """
        try:
            future = self.executor.submit(self._run_single_query)
            result = future.result(timeout=timeout)
            self.logger.info("Single query executed successfully")
            return result
        except TimeoutError:
            self.logger.error(f"Query execution timed out after {timeout} seconds")
            return self._get_default_probabilities()
        except Exception as e:
            self.logger.error(f"Single query execution failed: {e}")
            return self._get_default_probabilities()

    def _run_single_query(self) -> Dict[str, float]:
        """Internal method to run single query."""
        model = self._load_rules()
        return self._evaluate_queries(model)

    def run_batch_queries(self, num_queries: int = 5, timeout: int = 60) -> List[Dict[str, float]]:
        """
        Execute multiple ProbLog queries in batch.

        Args:
            num_queries (int): Number of queries to execute
            timeout (int): Total timeout for batch execution

        Returns:
            List[Dict[str, float]]: List of query results
        """
        try:
            future = self.executor.submit(self._run_batch_queries, num_queries)
            results = future.result(timeout=timeout)
            self.logger.info(f"Batch of {num_queries} queries executed successfully")
            return results
        except TimeoutError:
            self.logger.error(f"Batch query execution timed out after {timeout} seconds")
            return [self._get_default_probabilities() for _ in range(num_queries)]
        except Exception as e:
            self.logger.error(f"Batch query execution failed: {e}")
            return [self._get_default_probabilities() for _ in range(num_queries)]

    def _run_batch_queries(self, num_queries: int) -> List[Dict[str, float]]:
        """Internal method to run batch queries."""
        model = self._load_rules()
        return [self._evaluate_queries(model) for _ in range(num_queries)]

    def add_evidence(self, evidence: Dict[str, Any]) -> bool:
        """
        Add evidence to the ProbLog model.

        Args:
            evidence: Dictionary of evidence values

        Returns:
            bool: Success status
        """
        try:
            model = self._load_rules()
            for predicate, value in evidence.items():
                evidence_term = Term.from_string(f"evidence({predicate}, {value})")
                model.add_clause(evidence_term)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add evidence: {e}")
            return False

    def save_results(self, results: Union[Dict[str, float], List[Dict[str, float]]],
                     output_path: Optional[str] = None) -> str:
        """
        Save query results to JSON file.

        Args:
            results: Query results to save
            output_path: Optional custom path for results file

        Returns:
            str: Path where results were saved
        """
        try:
            if output_path is None:
                output_dir = 'results'
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(output_dir, f'prob_log_results_{timestamp}.json')

            # Prepare results with metadata
            save_data = {
                'results': results,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'rules_path': self.rules_path,
                    'num_queries': len(results) if isinstance(results, list) else 1
                }
            }

            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=4)

            self.logger.info(f"Results saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

    def _get_default_probabilities(self) -> Dict[str, float]:
        """Get default probability values for error cases."""
        return {
            'system_state_normal': 0.0,
            'system_state_degraded': 0.0,
            'system_state_critical': 0.0,
            'failure_risk': 0.0,
            'system_stress': 0.0,
            'efficiency_drop': 0.0,
            'abnormal_pattern': 0.0,
            'safety_critical': 0.0,
            'performance_degraded': 0.0
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        if exc_type:
            self.logger.error(f"Error during execution: {exc_val}")
            return False
        return True

    def get_interface_info(self) -> Dict[str, Any]:
        """Get information about the interface configuration."""
        return {
            'rules_path': self.rules_path,
            'available_queries': list(self._get_default_probabilities().keys()),
            'version': '2.2',  # ProbLog version
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    try:
        with ProbLogInterface() as prob_log:
            # Run single query
            single_result = prob_log.run_single_query()
            print("Single Query Result:", single_result)

            # Run batch queries
            batch_results = prob_log.run_batch_queries(num_queries=3)
            print("Batch Query Results:", batch_results)

            # Save results
            prob_log.save_results(batch_results)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)