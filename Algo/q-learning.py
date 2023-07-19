import numpy as np
import math
from collections import defaultdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gym
from gym import register
import os
import sys
import warnings
from tqdm import tqdm
from rich.console import Console
file_dir_name = os.path.dirname(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if file_dir_name not in sys.path:
    sys.path.append(file_dir_name)

from gym.envs.toy_text import FrozenLakeEnv
from env.simple_grid import DrunkenWalkEnv

# 可以将print输出美化
cs = Console()
warnings.filterwarnings('ignore')
cs.print(f"gym version == {gym.__version__}")