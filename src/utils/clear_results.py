# src/utils/clear_results.py
import os
import shutil

def clear_results(results_dir):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
