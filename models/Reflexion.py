# -*- coding:utf-8 _*-
from torch.utils.data import DataLoader, SequentialSampler
from utils import *
import time
from models import *
from torcheval.metrics import HitRate, ReciprocalRank
import torchmetrics
from dataSet import *
from tqdm import tqdm
from math import sqrt, log
from executors import AppsExecutor, HumanevalExecutor
from ChatModels import GPTChat
import re
import astroid
from astroid import nodes
from astroid.builder import AstroidBuilder
import dataclasses
from typing import List, Union, Optional, Literal
from .staticfg import CFGBuilder
import jsonlines

IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"

PY_SELF_REFLECTION_CHAT_INSTRUCTION = "You are a Python programming assistant. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation."

USE_PYTHON_CODEBLOCK_INSTRUCTION = ("Use a Python code block to write your response. "
                                    "Remember to contain the complete program including all the imports and function header in your response."
                                    "Answer with the code ONLY. No other explanation or words attached!")

PY_REFLEXION_FEW_SHOT_ADD = '''Example 1:
[previous impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

[unit test results from previous impl]:
Tested passed:

Tests failed:
assert add(1, 2) == 3 # output: -1
assert add(1, 2) == 4 # output: -1

[reflection on previous impl]:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.

[improved impl]:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a + b
```
'''

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def print_messages(messages: List[Message], prefix = "") -> None:
    print("::CHAT MESSAGE::" +prefix)
    for msg in messages:
        print(msg.content)
    print("==================")


class Reflexion:
    def __init__(self, args):
        self.args = args
        self.sample_nums = 0
        self.gamma = 0.9
        self.save_mid_json = []
        if 'gpt3.5' in args.arch or 'gpt4' in args.arch:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = GPTChat(args.arch, self.tokenizer, args, self.save_mid_json)
        else:
            raise ValueError("wrong chat model")

        if args.dataset == 'apps':
            self.executor = AppsExecutor(args)
        elif args.dataset == 'humaneval':
            self.executor = HumanevalExecutor(args)

        self.term_cond = lambda: self.sample_nums > args.max_sample_times

        self.cached_reward = {}
        self.cached_verbal_feedback = {}
        self.cur_prob_instance = None
        self.sample_times = []
        self.st = time.time()

    def generate(self, problem_instance):
        self.st = time.time()
        self.cur_prob_instance = problem_instance
        raw_prompt = problem_instance['prompt']
        done = False
        initial_state = self.tokenizer.encode(raw_prompt)
        if len(initial_state) >= self.args.horizon:
            return None

        cur_pass = 0
        pass_at_k = 1
        is_pass_train = False
        train_reward = 0.0
        test_reward = 0.0
        while cur_pass < pass_at_k and not is_pass_train:
            cur_iter = 0
            # first attempt
            code_id = self.generator.get_rationale_predicted_sequence(initial_state)
            cur_func_impl = self.tokenizer.decode(code_id)

            full_result = self.get_reward(code_id, with_verbal=True)
            complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

            if complete_prog_score >= 1:
                is_pass_train = True
                train_reward = complete_prog_score
                test_reward = self.get_reward(code_id, mode='test')
                break
            # use debug to iteratively improve
            failed_test_list = []
            failed_tests = ''
            tmp_count = 0
            for k, verbal_feedback in enumerate(verbal_feedbacks):
                if not isinstance(verbal_feedback, str):  # 有failed test情况下，verbal_feedback是dict而不是str
                    if tmp_count <= 5:
                        self.args.verbal_length_check_num += 1
                        if len(self.tokenizer.encode(verbal_feedback['output'], allowed_special={'<|endoftext|>'})) > 2048:
                            self.args.verbal_length_exd_num += 1
                            tmp_shorter = self.tokenizer.encode(verbal_feedback['output'], allowed_special={'<|endoftext|>'})[:2048]
                            verbal_feedback['output'] = self.tokenizer.decode(tmp_shorter)
                        failed_tests += f"\n\n## Failed test {tmp_count + 1}: {verbal_feedback['output']}"
                        failed_test_list.append(verbal_feedback['output'])
                        tmp_count += 1
            messages = []
            max_iters = 10
            cur_state = self.tokenizer.decode(initial_state)
            while cur_iter < max_iters:
                print('---------------')
                print('cur_iter:', cur_iter)
                print('---------------')

                reflection = self.gen_reflection(cur_state, cur_func_impl, failed_tests)

                cur_func_impl = self.func_impl(cur_state=cur_state,
                                               prev_func_impl=cur_func_impl,
                                               cur_feedback=failed_tests,
                                               reflection=reflection)

                code_id = self.tokenizer.encode(cur_func_impl)
                full_result = self.get_reward(code_id, with_verbal=True)
                complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

                failed_test_list = []
                failed_tests = ''
                tmp_count = 0
                for k, verbal_feedback in enumerate(verbal_feedbacks):
                    if not isinstance(verbal_feedback, str):  # 有failed test情况下，verbal_feedback是dict而不是str
                        if tmp_count <= 5:
                            self.args.verbal_length_check_num += 1
                            if len(self.tokenizer.encode(verbal_feedback['output'], allowed_special={'<|endoftext|>'})) > 2048:
                                self.args.verbal_length_exd_num += 1
                                tmp_shorter = self.tokenizer.encode(verbal_feedback['output'], allowed_special={'<|endoftext|>'})[:2048]
                                verbal_feedback['output'] = self.tokenizer.decode(tmp_shorter)
                            failed_tests += f"\n\n## Failed test {tmp_count + 1}: {verbal_feedback['output']}"
                            failed_test_list.append(verbal_feedback['output'])
                            tmp_count += 1

                if complete_prog_score == 1.0 or cur_iter == max_iters - 1:
                    is_pass_train = True
                    cur_iter += 1
                    train_reward = complete_prog_score
                    test_reward = self.get_reward(code_id, mode='test')
                    break
                cur_iter += 1
            cur_pass += 1

        # original mcts part
        complete_programs_ids = list(map(lambda x: list(x), self.cached_reward.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_reward[tuple(s)] for s in complete_programs_ids]
        test_rewards = [self.get_reward(s, mode='test') for s in complete_programs_ids]

        output_dict = {}
        output_dict['final_program'] = cur_func_impl
        output_dict['train_reward'] = train_reward  # 这里的train reward对应的是test的code的
        output_dict['test_reward'] = test_reward  # 这里的test reward并不是最高train reward的，而是最后一次的train reward的代码
        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        self.cached_reward = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.sample_nums = 0
        self.save_mid_json = []
        self.generator.save_mid_json = self.save_mid_json
        self.args.rollout_count = -1

        return output_dict

    def get_reward(self, s, mode='train', with_verbal=False):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            if with_verbal:
                return [self.cached_reward[tuple(s)], self.cached_verbal_feedback[tuple(s)]]
            else:
                return self.cached_reward[tuple(s)]

        # 转换成文本
        output_str = self.convert_state_to_program(s)

        # 计算pass rate
        try:
            curr_res = self.executor.check_correctness(self.cur_prob_instance, output_str, mode, with_verbal=with_verbal)  # with_verbal: curr_res=[[True/False, feedback_dict]]
            fixed = []
            verbal_feedbacks = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                if with_verbal:
                    verbal_feedbacks.append(e[1])
                    e = e[0]
                fixed.append(e)

            curr_res = fixed
            # if not np.all(curr_res):
            #     print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            curr_res = []

        # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
        assert isinstance(curr_res, list)
        pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
        reward = pass_rate

        # 添加到cached reward
        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
            if with_verbal:
                self.cached_verbal_feedback[tuple(s)] = verbal_feedbacks

        if with_verbal:
            return [reward, verbal_feedbacks]
        else:
            return reward

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s
    def gen_reflection(self, cur_state, cur_func_impl, failed_tests):
        verbalFeedback = (f"{cur_state}\n\n{cur_func_impl}\n\n"
                          f"The code generated following the thoughts doesn't pass some test cases. Here are the test cases the code doesn't pass: \n"
                          f"{failed_tests} \n")
        input_prompt = verbalFeedback + f"\nPlease provide a short reflection in two sentences on the code and errors. This reflection should remind the programmer not to make the mistake.\n" \

        print('\n--------------7 reflection input prompt')
        print(input_prompt)

        response, _ = self.generator.generate_response_api(input_prompt,
                                                           top_k=1,
                                                           max_length=1024,
                                                           system_message=PY_SELF_REFLECTION_CHAT_INSTRUCTION)

        print('\n--------------8 reflection response')
        print(response)
        response = f"Reflection: {response}"
        return response

    def func_impl(self, cur_state, prev_func_impl, cur_feedback, reflection):
        message = f"{PY_REFLEXION_FEW_SHOT_ADD}\n[previous impl]:\n{prev_func_impl}\n\n[unit test results from previous impl]:\n{cur_feedback}\n\n[reflection on previous impl]:\n{reflection}\n\n[improved impl]:\n{cur_state}"
        prompt = f"{PY_SELF_REFLECTION_CHAT_INSTRUCTION}\n{USE_PYTHON_CODEBLOCK_INSTRUCTION}"
        # func_bodies is a really bad name, as it can also be just 1 string

        messages = [
            Message(
                role="system",
                content=prompt,
            ),
            Message(
                role="user",
                content=PY_REFLEXION_FEW_SHOT_ADD,
            ),
            Message(
                role="assistant",
                content='\n[previous impl]:\n' + prev_func_impl,
            ),
            Message(
                role="user",
                content=f"[unit test results from previous impl]:\n{cur_feedback}\n\n[reflection on previous impl]:",
            ),
            Message(
                role="assistant",
                content=reflection,
            ),
            Message(
                role="user",
                content=f"[improved impl]:\n{cur_state}",
            ),
        ]

        print_messages(messages)

        func_bodies = self.generator.generate_chat(messages=messages,
                                                   num_comps=1,
                                                   temperature=0,
                                                   stop=['[debug end]', 'Here is the updated code:'])
        return func_bodies