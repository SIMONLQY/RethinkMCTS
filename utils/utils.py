# -*- coding:utf-8 _*-
import os
import torch
import copy
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
import transformers
import tiktoken
import re
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')


def get_raw_data_path():
    return '/home/xxx'


def get_proj_path(proj_name='RethinkMCTS'):
    """
    :param item_name: project name
    :return:
    """
    cur_path = os.path.abspath(os.path.dirname(__file__))

    return cur_path[:cur_path.find(proj_name)] + proj_name


def extract_generated_test_cases(text):
    pattern = re.compile(r"assert (\w+)(\(.*?\)) (==|<=|>=|<|>)\s*(.*)")
    matches = re.findall(pattern, text)

    test_cases = []
    for function_name, inputs, operator, output in matches:
        case = f"assert {function_name}({inputs}) {operator} {output}"
        test_cases.append(case)

    return test_cases


if __name__ == '__main__':
    example_text = """
**1. Basic Test Cases**:

```python
# Basic test case with normal input
assert numerical_letter_grade([4.0, 3, 1.7, 2, 3.5]) == ['A+', 'B', 'C-', 'C', 'A-']

# Test case with all grades being the minimum value
assert numerical_letter_grade([0.0, 0.0, 0.0]) == ['E', 'E', 'E']

# Test case with all grades being the maximum value
assert numerical_letter_grade([4.0, 4.0, 4.0]) == ['A+', 'A+', 'A+']
```

**2. Edge Test Cases**:

```python
# Test case with an empty list of grades
assert numerical_letter_grade([]) == []

# Test case with negative GPA values
assert numerical_letter_grade([-1.0, -2.5, -3.7]) == ['E', 'E', 'E']

# Test case with a mix of negative and positive GPA values
assert numerical_letter_grade([-1.0, 3.5, -2.0, 4.0]) == ['E', 'A-', 'E', 'A+']

# Test case with a GPA value on the threshold
assert numerical_letter_grade([3.0, 2.7, 2.3, 2.0, 1.7, 1.3, 1.0, 0.7, 0.0]) == ['B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-']
```

**3. Large Scale Test Cases**:

```python
# Test case with a large number of grades
assert numerical_letter_grade([3.5] * 1000) == ['A-'] * 1000

# Test case with a large range of GPA values
assert numerical_letter_grade([i/10 for i in range(0, 41)]) == ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'D-', 'D-', 'D-', 'D-', 'D-', 'D-', 'D-', 'D-', 'D-', 'D-', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D+', 'D+', 'D+', 'D+', 'D+', 'D+', 'D+', 'D+', 'D+', 'D+']
```
    """

    test_cases_list = extract_generated_test_cases(example_text)

    for case in test_cases_list:
        print(case)



# def eval_and_save_problems(dataset ,exp_id):
#     """
#     Args:
#         args: command arguments
#         indices: the indices of problems to be evaluated
#     """
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Get experiments results")
#     parser.add_argument("--save", type=str, default=f"{get_proj_path()}/results/{dataset}/Experiment_{exp_id}", help="Where the evaluated data is loaded from and results saved to.")
#     parser.add_argument("-n", default=1, type=int, help='Evaluate using the n best program candidates (the n programs that have the highest pass rate on the training set.')
#     parser.add_argument('--retest', action='store_true', default=False, help="rerun tests.")
#
#     args = parser.parse_args()
#
#
#     files = os.listdir(args.save)
#     indices = [int(re.findall(r'\d+', file)[0]) for file in files if (file.endswith('.json') and 'result' not in file)]
#     all_results_loc = os.path.join(args.save, f"all_results.json")
#     # try:
#     #     with open(all_results_loc, 'r') as f:
#     #         all_results = json.load(f)
#     #
#     #     rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
#     #         all_results['rewards'], all_results['rewards_train'], all_results['times'], all_results['sample_times'], all_results['compile_errors'], all_results['runtime_errors']
#     # except:
#     #     print(f"{all_results_loc} specified, but failed to open.")
#     rewards, rewards_train, times, sample_times, compile_errors, runtime_errors, codes = {}, {}, {}, {}, {}, {}, {}
#     input_token_nums = {}
#     output_token_nums = {}
#
#     # don't show progress bar if only one problem to test
#     indices = tqdm(indices) if len(indices) > 1 else indices
#
#     for index in indices:
#         if str(index) in rewards.keys() and not args.retest:
#             # print(f"skipping {index} because it's already in results json file")
#             continue
#
#         code_loc = os.path.join(args.save, f"{index}.json")
#
#         if not os.path.exists(code_loc):
#             # print(f"didn't find result for {index}")
#             continue
#         else:
#             # result_loc exists, simply read it
#             try:
#                 with open(code_loc) as f:
#                     result = json.load(f)
#
#                     reward, reward_train, time, sample_time = result['rewards'], result['train rewards'], result['time'], result['sample times']
#                     if 'input_token_num' in result.keys():
#                         input_token_nums[index] = int(result['input_token_num'])
#                     if 'output_token_num' in result.keys():
#                         output_token_nums[index] = int(result['output_token_num'])
#             except Exception as e:
#                 print(f"failed to read {code_loc}, {e}")
#                 continue
#
#         if len(reward) == 0 or (isinstance(time, list) and len(time) == 0):
#             print(f"{code_loc} results non-parsable.")
#             continue
#
#         if len(reward_train) > 0:
#             # sort the training rewards of the samples, get the top n of them
#             top_n_indices = np.argsort(reward_train)[::-1][:args.n]  # arges.n = 1
#             # find the one that has the highest test reward
#             return_index = max(top_n_indices, key=lambda x: reward[x])
#         else:
#             return_index = 0
#
#         # add to the list
#         rewards[index] = reward[return_index]
#
#         # get best-possible result
#         # if 1.0 in reward:
#         #     rewards[index] = 1.0
#         # else:
#         #     rewards[index] = reward[return_index]
#
#         codes[index] = result['codes']
#
#         rewards_train[index] = reward_train[return_index] if len(reward_train) > 0 else 0
#         # these values are None for failed experiments
#         if time is not None: times[index] = time
#
#         sample_times[index] = sample_time
#
#         try:
#             compile_errors[index] = result['compile_error']
#             runtime_errors[index] = result['runtime_error']
#         except:
#             compile_errors[index] = 0
#             runtime_errors[index] = 0
#
#     # save results to file
#     all_results = {
#         'rewards': rewards,
#         'rewards_train': rewards_train,
#         'times': times,
#         'sample_times': sample_times,
#         'compile_errors': compile_errors,
#         'runtime_errors': runtime_errors,
#         'codes': codes,
#         'input_token_nums': input_token_nums,
#         'output_token_nums': output_token_nums,
#     }
#
#     with open(all_results_loc, "w") as f:
#         try:
#             json.dump(all_results, f)
#         except Exception as e:
#             print(f"Couldn't save all results.\n{e}")
#
#     # return results from args.start to args.end
#     filter_op = lambda x: {k: x[k] for k in x.keys() if int(k) in indices}
#     ret_results = {k: filter_op(v) for k, v in all_results.items()}
#
#     return ret_results