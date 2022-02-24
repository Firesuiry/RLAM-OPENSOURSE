import os
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TASK_PATH = Path('data/task/')
CACHE_PATH = Path('data/cache/')
if not os.path.exists(TASK_PATH):
    os.makedirs(TASK_PATH)
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)
