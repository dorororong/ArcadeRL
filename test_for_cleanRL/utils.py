# utils.py
import os
from datetime import datetime

def make_log_dir(base="logs", algo_name="run"):
    """
    base/20250514_150305_algo_name 형태의 폴더를 만들고 경로를 반환합니다.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"{now}_{algo_name}")
    os.makedirs(path, exist_ok=True)
    return path
