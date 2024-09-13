import numpy as np
import torch
from utils import *
import os
import json
import jsonlines


class HumanevalHandler:
    def __init__(self, problem_indices, args):
        problem_file = f"{get_proj_path()}/dataProcess/humaneval/processed.jsonl"
        self.problems = []
        with jsonlines.open(problem_file) as reader:
            for item in reader:
                if int(item['task_id'].split('/')[1]) in problem_indices:
                    self.problems.append(item)
