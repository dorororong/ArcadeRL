# utils.py
import os
import datetime

def make_log_dir(algo_name="experiment", base_dir="logs"):
    """
    Creates a timestamped directory for logs.
    Example: logs/my_algo_2023-10-27_15-30-00/
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_name = f"{algo_name}_{timestamp}"
    full_log_dir = os.path.join(base_dir, log_dir_name)
    
    # os.makedirs(full_log_dir, exist_ok=True) # This is handled in dueling.py
    return full_log_dir