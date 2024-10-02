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
from .ldb import ldb_debug
import json


def uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.ucb)


def p_uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.p_ucb)


def var_p_uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.var_p_ucb)


class LATS:
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
        if args.uct_alg == 'uct':
            self.node_choose_policy = uct_tree_policy
        elif args.uct_alg == 'p_uct':
            self.node_choose_policy = p_uct_tree_policy
        elif args.uct_alg == 'var_p_uct':
            self.node_choose_policy = var_p_uct_tree_policy
            self.ucb_base = args.ucb_base
        self.root = None
        self.cached_reward = {}
        self.cached_value = {}
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

        if self.args.json_save_all:
            if self.args.dataset == 'humaneval':
                self.save_mid_json.append(f"ques_id: \n{problem_instance['task_id'].split('/')[-1]}")
                self.save_mid_json.append(f"given_tests: \n{problem_instance['given_tests']}")
                self.save_mid_json.append(f"tests: \n{problem_instance['test']}")

        self.mcts_procedure(initial_state, done)

        if len(self.cached_value) == 0:
            state = self.generator.get_code_predicted_sequence(initial_state)
            complete_prog_score = self.get_reward(state)

        complete_programs_ids = list(map(lambda x: list(x), self.cached_value.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_value[tuple(s)] for s in complete_programs_ids]
        test_rewards = [self.get_reward(s, mode='test') for s in complete_programs_ids]
        best_idx = np.argmax(train_rewards)

        output_dict = {}
        output_dict['final_program'] = complete_programs[best_idx]
        output_dict['train_reward'] = train_rewards[best_idx]
        output_dict['test_reward'] = test_rewards[best_idx]

        if self.args.json_save_all:
            self.save_mid_json.append(f"final_program: \n{complete_programs[best_idx]}")
            self.save_mid_json.append(f"test_reward: \n{test_rewards[best_idx]}")
            result_loc = f"{get_proj_path()}/results/{self.args.dataset}/Experiment_{self.args.experiment_idx}/middle_process/"
            os.makedirs(result_loc, exist_ok=True)
            result_loc = os.path.join(result_loc, f"{int(problem_instance['task_id'].split('/')[-1])}.json")
            with open(result_loc, "w") as f:
                json.dump(self.save_mid_json, f)

        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        self.cached_reward = {}
        self.cached_value = {}
        self.cached_verbal_feedback = {}
        self.generator.clean_cache()
        self.sample_nums = 0
        self.save_mid_json = []

        return output_dict

    def mcts_procedure(self, initial_state, done):
        """
        Compute the entire MCTS procedure wrt to the selected tree policy.
        Function tree_policy is a function taking an agent + a list of ChanceNodes as argument
        and returning the one chosen by the tree policy.
        """
        decision_node_num = 0
        self.root = MineDecisionNode(None, initial_state, done, generator=self.generator, id=decision_node_num, tokenizer=self.tokenizer, initial_state=initial_state)
        self.root.__expand__()
        decision_node_num += 1
        print("Performing rollouts.")
        for rollout_count in range(self.args.rollout):
            self.args.rollout_count = rollout_count
            if self.term_cond():
                break
            rewards = []  # Rewards collected along the tree for the current rollout
            node = self.root  # Current node
            terminal = done

            # Selection
            select = True
            while select:
                if (type(node) == MineDecisionNode):  # DecisionNode
                    if node.is_terminal:
                        select = False  # Selected a terminal DecisionNode
                    else:
                        node = self.node_choose_policy(self, node.children)
                else:
                    state_p, reward, terminal = self.transition(node.parent.state, node.action)
                    rewards.append(reward)

                    new_state = True
                    for i in range(len(node.children)):
                        if node.children[i].state == state_p:
                            node = node.children[i]
                            new_state = False
                            break
                    if new_state:
                        select = False  # Selected a ChanceNode

            selection_print = []
            tmp_node = node
            selection_print.append(f'{tmp_node.id}')
            while tmp_node.parent.parent:
                tmp_node = tmp_node.parent.parent
                selection_print.append(f'{tmp_node.id}')
            tmp_text = '->'.join(selection_print[::-1])
            print(f"selection_{self.args.rollout_count}: \n{'root->' + tmp_text}")
            # print('root->' + tmp_text)
            if self.args.json_save_all:
                self.save_mid_json.append(f"selection_{self.args.rollout_count}: \n{'root->' + tmp_text}")

            # Expansion
            # If node is a decision node, then it must be a terminal node, do nothing here
            if type(node) == MineChanceNode:
                # print('\n-----------1selected action: ')
                # print(f"{self.tokenizer.decode([node.action])}")

                node.children.append(MineDecisionNode(node, state_p, terminal, generator=self.generator, id=decision_node_num, decision_memory=node.chance_memory, tokenizer=self.tokenizer, initial_state=initial_state))
                decision_node_num += 1
                node = node.children[-1]  # 就是新增加的decision node

            # Evaluation
            # now `rewards` collected all rewards in the ChanceNodes above this node
            assert (type(node) == MineDecisionNode)

            # no reflect
            state = node.get_with_reflection_state(initial_state)
            if not node.is_terminal:
                """
                verbal feedback that is used as memory for the node expansion
                """
                code_id = self.generator.get_code_predicted_sequence(state)

                full_result = self.get_reward(code_id, with_verbal=True)
                complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

                if self.args.json_save_all:
                    self.save_mid_json.append(f'given_test_output_reward_{self.args.rollout_count}: \n{complete_prog_score}')

                # value estimation
                estimate = complete_prog_score
                ###################

                if tuple(code_id) not in self.cached_value.keys():
                    self.cached_value[tuple(code_id)] = estimate

                failed_tests = ''
                verbalFeedback = ''
                failed_test_list = []
                tmp_count = 0
                for k, verbal_feedback in enumerate(verbal_feedbacks):
                    if not isinstance(verbal_feedback, str):
                        if tmp_count <= 5:
                            self.args.verbal_length_check_num += 1
                            if len(self.tokenizer.encode(verbal_feedback['output'], allowed_special={'<|endoftext|>'})) > 2048:
                                self.args.verbal_length_exd_num += 1
                                tmp_shorter = self.tokenizer.encode(verbal_feedback['output'], allowed_special={'<|endoftext|>'})[:2048]
                                verbal_feedback['output'] = self.tokenizer.decode(tmp_shorter)
                            failed_tests += f"\n\n## Failed test {tmp_count + 1}: {verbal_feedback['output']}"
                            failed_test_list.append(verbal_feedback['output'])
                            tmp_count += 1
                if failed_tests != '':
                    code = self.tokenizer.decode(code_id)

                    system_msg = f"You are an expert in programming."
                    verbalFeedback = (f"{self.tokenizer.decode(state)}\n\n[Current impl]:\n{code}\n\n"
                                      f"Above is the implementations of previous codes and current implementation of codes. The current code is generated based on the previous codes. "
                                      f"However, the current code generated doesn't pass some test cases. Here are the test cases the code doesn't pass: \n"
                                      f"{failed_tests} \n")

                    self.args.verbal_length_check_num += 1
                    if len(self.tokenizer.encode(verbalFeedback, allowed_special={'<|endoftext|>'})) > 12000:
                        self.args.verbal_length_exd_num += 1
                        tmp_shorter = self.tokenizer.encode(verbalFeedback, allowed_special={'<|endoftext|>'})[:8000]
                        verbalFeedback = self.tokenizer.decode(tmp_shorter)

                    # no reflect
                    input_prompt = verbalFeedback + f"\nPlease provide a short reflection in two sentences on the code and errors. This reflection should remind the programmer not to make the mistake.\n" \

                    print('\n--------------7 summarizing input prompt')
                    print(input_prompt)

                    response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024, system_message=system_msg)

                    print('\n--------------8 summarizing response')
                    print(response)
                    memory = f"\n[Unit test results from this impl]: \n{failed_tests}\n\n[Reflection on this previous impl]: \n{response}"

                    node.decision_memory = self.tokenizer.encode(memory)

                    node.__expand__(verbal_feedback=verbalFeedback)
                else:
                    # no reflect
                    memory = f"[Unit test result]: All visible test cases passed for this implementation."
                    node.decision_memory = self.tokenizer.encode(memory)
                    node.__expand__()

                # save this information for demo
                node.info['complete_program'] = code_id

                self.sample_nums = self.sample_nums + 1
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                estimate = 0

            # Backpropagation
            # Backpropagation of scaled reward
            node.visits += 1
            node = node.parent
            assert (type(node) == MineChanceNode)
            while node:
                if len(rewards) != 0:
                    estimate = rewards.pop() + self.gamma * estimate
                node.sampled_returns.append(estimate)
                node.parent.visits += 1
                node = node.parent.parent

            # should finish backpropagating all the rewards back
            assert len(rewards) == 0
            self.sample_times.append(time.time() - self.st)

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s

    def get_reward(self, s, mode='train', with_verbal=False):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            if with_verbal:
                return [self.cached_reward[tuple(s)], self.cached_verbal_feedback[tuple(s)]]
            else:
                return self.cached_reward[tuple(s)]

        output_str = self.convert_state_to_program(s)

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

        if mode == 'train':
            self.cached_reward[tuple(s)] = reward
            if with_verbal:
                self.cached_verbal_feedback[tuple(s)] = verbal_feedbacks

        if with_verbal:
            return [reward, verbal_feedbacks]
        else:
            return reward

    def transition(self, s, a):
        if isinstance(a, list):
            next_state = s + a
        else:
            next_state = s + [a]
        if self.generator.terminal_token in a or len(next_state) == self.args.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(next_state)
            if tuple(next_state) not in self.cached_value.keys():
                self.cached_value[tuple(next_state)] = reward
        else:
            reward = 0  # no intermediate reward
        return next_state, reward, done

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return chance_node_value(node) + self.args.ucb_constant * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, weighted by prior probability
        """
        return chance_node_value(node) + self.args.ucb_constant * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def var_p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, the ucb exploration weight is a variable
        """
        ucb_parameter = log((node.parent.visits + self.ucb_base + 1) / self.ucb_base) + self.args.ucb_constant
        return chance_node_value(node) + ucb_parameter * node.prob * sqrt(log(node.parent.visits)) / (1 + len(node.sampled_returns))

    def gen_test_evaluation(self, cur_state, cur_code=None, initial_state=None):
        self.args.generate_tests_total += 1

        system_msg = f"As a tester, your task is to create comprehensive test cases for the incomplete Python function provided below."
        input_prompt = \
f"""
*Role**: As a tester, your task is to create comprehensive test cases for the incomplete Python function provided below. These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's 
robustness, reliability, and scalability.
**Input Code Snippet**:
```python
{self.tokenizer.decode(initial_state)}
```
**1. Basic Test Cases**:
- **Objective**: To verify the fundamental functionality of the `has_close_elements` function under normal
conditions.
**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.
**3. Large Scale Test Cases**:
- **Objective**: To assess the function’s performance and scalability with large data samples.
**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.
- You should response with the test cases only, not the code implementation. Each test case should be an assert expression.
"""
        print('\n--------------14 evaluation generate tests prompt')
        print(input_prompt)
        response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=2048, system_message=system_msg)

        print('\n--------------15 evaluation generate tests output')
        print(response)
        test_case_list = extract_generated_test_cases(response)
        self.cur_prob_instance['generated_tests'] = test_case_list
        with_verbal = False

        try:
            if len(test_case_list) == 0:
                assert False

            curr_res = self.executor.check_correctness(self.cur_prob_instance, cur_code, mode='generated_tests', with_verbal=with_verbal)  # with_verbal: curr_res=[[True/False, feedback_dict]]
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

            # How to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
            assert isinstance(curr_res, list)
            pass_rate = np.mean(np.asarray(curr_res) > 0) if len(curr_res) > 0 else 0
            reward = pass_rate
            return reward
        except Exception as e:
            self.args.failed_generate_tests_count += 1
            print(f"test framework exception = {repr(e)}{e}\n")
            input_prompt = (f"{self.tokenizer.decode(cur_state)}\n\n{cur_code}\n\n"
                            f"Above is a Python code problem with the thoughts and code to solve the problem. The code could pass all the example test cases, however, it may or may not be completely correct. \n"
                            f"Please evaluate and return the correctness score in range [-1, 1]\n"
                            f"Evaluate the correctness of the code and give only ONE evaluation score. \n"
                            f"The code's correctness is whether it can pass all the possible unseen test cases of the problem, not just the given ones."
                            f"Example Answers: \n"
                            f"{{\"evaluation\": -0.5,  \"explanation\": \"The code is far from correct for solving the problem.\"}} \n"
                            f"{{\"evaluation\": 1.0, \"explanation\": \"The generated code is the correct solution that can pass all the possible test cases and strange corner cases too. \"}} \n"
                            f"{{\"evaluation\": 0.1, \"explanation\": \"The code is not the correct solution but can pass some simple test cases. \"}} \n")
            print('\n--------------5 evaluation input prompt')
            print(input_prompt)
            if self.args.json_save_all:
                self.save_mid_json.append(f'lm_eval_input_{self.args.rollout_count}: \n{input_prompt}')

            response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024, system_message=system_msg)

            print('\n--------------6 evaluation response')
            print(response)
            try:
                float_pattern = re.compile(r'-?\d+\.?\d*')
                response_scores = [float(match) for match in float_pattern.findall(response)]
                evaluation = response_scores[0]
            except Exception as e:
                print(f"Error in parsing evaluation response: {repr(e)}{e}")
                evaluation = 0.0
            return evaluation


class MineDecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """

    def __init__(self, parent, state, is_terminal=False, generator=None, id=None, decision_memory=[], tokenizer=None, initial_state=''):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        self.generator = generator
        self.tokenizer = tokenizer
        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}
        self.decision_memory = decision_memory
        self.initial_state = initial_state
        self.second_chance_flag = True

    def __expand__(self, verbal_feedback=''):
        if verbal_feedback == '':
            expand_prompt_id = self.get_with_reflection_state(initial_state=self.initial_state)

            # no reflect
            # expand_prompt_id = self.state
            top_k_line_predict, top_k_scores = self.generator.get_top_k_codes_predict(expand_prompt_id)
        else:
            expand_prompt_id = self.generator.tokenizer.encode(verbal_feedback)
            top_k_line_predict, top_k_scores = self.generator.get_top_k_codes_predict(expand_prompt_id, with_verbal=True)

        self.possible_actions = top_k_line_predict
        self.action_scores = top_k_scores

        # populate its children
        self.children = [MineChanceNode(self, (act, score), chance_memory=self.decision_memory, id=id) for id, (act, score) in enumerate(zip(self.possible_actions, self.action_scores))]

        # print('\n---------------2children tokens:')
        # children_tokens = []
        # for child_token in self.possible_actions:
        #     children_tokens.append(generator.tokenizer.decode([child_token]))
        # print(f"{children_tokens}")
    #
    # def __reflect__(self, new_experience=''):
    #     system_msg = f"You are an expert programmer."
    #     input_prompt = (f"Based on your previous reflection and the new experience, please provide a new reflection on the code and errors. This reflection should remind the programmer not to make the mistake."
    #                     f"The reflection should be short within two sentences. \n"
    #                     f"Previous reflection: {self.decision_memory}\n"
    #                     f"New experience: {new_experience}\n")
    #
    #     response, _ = self.generator.generate_response_api(input_prompt, top_k=1, max_length=1024, system_message=system_msg)
    #     if 'Reflection:' not in response:
    #         response = 'Reflection: ' + response
    #     response = '\n' + response + '\n'
    #     self.decision_memory = self.tokenizer.encode(response)

    def get_with_reflection_state(self, initial_state):
        # one by one way
        rst_state = initial_state
        code_test_reflections = []
        if self.parent:
            immediate_mem = self.decision_memory
            code_test_reflections.append(immediate_mem)
        cur_node = self.parent


        trace_node_num = 1
        tmp_cur_node = self.parent
        while tmp_cur_node:
            tmp_cur_node = tmp_cur_node.parent.parent
            trace_node_num += 1

        count = trace_node_num - 1
        while cur_node:
            code_test_reflection_pair = cur_node.parent.decision_memory + self.tokenizer.encode(f'\n\n[Previous impl {count}]:\n') + cur_node.action
            code_test_reflections.append(code_test_reflection_pair)
            cur_node = cur_node.parent.parent
            count -= 1

        if len(code_test_reflections) > 0:
            for code_ref in code_test_reflections[::-1]:
                rst_state = rst_state + code_ref

        return rst_state

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class MineChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent, action_and_score, chance_memory=[], id=None):
        self.parent = parent
        self.action = action_and_score[0]
        self.depth = parent.depth
        self.children = []
        self.prob = action_and_score[1]  # the probability that this action should be token, provided by default policy
        self.sampled_returns = []
        self.chance_memory = chance_memory
        self.id = id

    def expanded(self):
        return len(self.children) > 0


def chance_node_value(node, mode="best"):
    """
    Value of a chance node
    """
    if len(node.sampled_returns) == 0:
        return 0

    if mode == "best":
        # max return (reasonable because the model is deterministic?)
        return max(node.sampled_returns)
    elif mode == "sample":
        # Use average return
        return sum(node.sampled_returns) / len(node.sampled_returns)
    else:
        raise Exception(f"Unknown tree search mode {mode}")
