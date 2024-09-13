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


def uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.ucb)


def p_uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.p_ucb)


def var_p_uct_tree_policy(mcts_agent, children):
    return max(children, key=mcts_agent.var_p_ucb)


class MCTSLine:
    def __init__(self, args):
        self.args = args
        self.sample_nums = 0
        self.gamma = 0.9
        if 'gpt3.5' in args.arch or 'gpt4' in args.arch:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.generator = GPTChat(args.arch, self.tokenizer, args)
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

        self.mcts_procedure(initial_state, done)

        if len(self.cached_reward) == 0:
            state = self.generator.get_line_predict_sequence(initial_state)
            complete_prog_score = self.get_reward(state)

        complete_programs_ids = list(map(lambda x: list(x), self.cached_reward.keys()))
        if complete_programs_ids is None or len(complete_programs_ids) == 0:
            return None
        complete_programs = [self.convert_state_to_program(s) for s in complete_programs_ids]

        train_rewards = [self.cached_reward[tuple(s)] for s in complete_programs_ids]
        test_rewards = [self.get_reward(s, mode='test') for s in complete_programs_ids]
        best_idx = np.argmax(train_rewards)

        output_dict = {}
        output_dict['final_program'] = complete_programs[best_idx]
        output_dict['train_reward'] = train_rewards[best_idx]
        output_dict['test_reward'] = test_rewards[best_idx]

        output_dict['all_programs'] = complete_programs
        output_dict['all_train_rewards'] = train_rewards
        output_dict['all_test_rewards'] = test_rewards
        output_dict['avg_sample_time'] = np.mean(np.array(self.sample_times))

        self.cached_reward = {}
        self.generator.clean_cache()
        self.sample_nums = 0

        return output_dict

    def mcts_procedure(self, initial_state, done):
        """
        Compute the entire MCTS procedure wrt to the selected tree policy.
        Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
        and returning the one chosen by the tree policy.
        """
        # 开始时，root是None
        decision_node_num = 0
        self.root = MineDecisionNode(None, initial_state, done, generator=self.generator, id=decision_node_num)
        decision_node_num += 1
        # 如果rollouts=1，产生的程序存在cached_rewards里面的只有一个完整程序，其实select那一步就已经选了走哪个完整程序了
        print("Performing rollouts.")
        for _ in range(self.args.rollout):  # 这个rollout控制的是选择次数，如果从根节点开始，第一次选第一层，第二次可能选的是第二层，第三次选第三层
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
                        node = self.node_choose_policy(self, node.children)  # 根据P-UCB从node的children中选择一个最大值的node， node is now a ChanceNode
                else:  # ChanceNode，（状态，动作）节点，相当于树中的一条边
                    state_p, reward, terminal = self.transition(node.parent.state, node.action)
                    rewards.append(reward)  # 做完动作没有terminal的情况下，reward为0，后面backpropagation主要靠estimation

                    new_state = True  # 如果树有很多层，这里的while循环会从根节点一层一层往下走，直到找到一个新的state_p
                    for i in range(len(node.children)):  # 其实chancenode只有一个child
                        if node.children[i].state == state_p:
                            # Shun: state_p already in the tree, point node to the corresponding Decision Node
                            node = node.children[i]
                            new_state = False
                            break
                    if new_state:  # 一开始如果是三个rollouts，就三个root的children都会经过这里
                        select = False  # Selected a ChanceNode

            # Expansion
            # If node is a decision node, then it must be a terminal node, do nothing here
            if type(node) == MineChanceNode:
                # print('\n-----------1selected action: ')
                # print(f"{self.tokenizer.decode([node.action])}")

                node.children.append(MineDecisionNode(node, state_p, terminal, generator=self.generator, id=decision_node_num))  # chance node 只有一个子节点，就是加上了那个动作的节点,但每一个decision node在创建的时候都会带有3个可能的动作
                decision_node_num += 1
                node = node.children[-1]  # 就是新增加的decision node

            # Evaluation
            # now `rewards` collected all rewards in the ChanceNodes above this node
            assert (type(node) == MineDecisionNode)
            state = node.state
            if not node.is_terminal:
                # follow the default policy to get a terminal state
                # print('\n--------3input prompt')
                # print(self.tokenizer.decode(state))

                state = self.generator.get_line_predict_sequence(state)

                # print('\n------4output sequence')
                # print(self.tokenizer.decode(state))


                estimate = self.get_reward(state)  # 这里的state包含了输入的prompt。在get reward这步会将state cache起来
                self.sample_nums = self.sample_nums + 1

                # save this information for demo
                node.info['complete_program'] = state  # decision node的info里面存了这个节点的可能的complete_program
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                estimate = 0

            # Backpropagation
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
        # root的children是chance node，每个对应于一个动作

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s

    def get_reward(self, s, mode='train'):
        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]

        # 转换成文本
        output_str = self.convert_state_to_program(s)

        # 计算pass rate
        try:
            curr_res = self.executor.check_correctness(self.cur_prob_instance, output_str, mode)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
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


class MineDecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """

    def __init__(self, parent, state, is_terminal=False, generator=None, id=None):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1

        # top_k_line_predict, top_k_scores = generator.get_top_k_line_predict(self.state)
        top_k_line_predict, top_k_scores = generator.get_top_k_line_predict_2(self.state)

        self.possible_actions = top_k_line_predict
        self.action_scores = top_k_scores

        # populate its children
        self.children = [MineChanceNode(self, (act, score)) for act, score in zip(self.possible_actions, self.action_scores)]

        # print('\n---------------2children tokens:')
        # children_tokens = []
        # for child_token in self.possible_actions:
        #     children_tokens.append(generator.tokenizer.decode([child_token]))
        # print(f"{children_tokens}")

        self.explored_children = 0
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits = 1
        # used to save any information of the state
        # we use this for saving complete programs generated from it
        self.info = {}

    def is_fully_expanded(self):
        return all([child.expanded() for child in self.children])


class MineChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent, action_and_score):
        self.parent = parent
        self.action = action_and_score[0]
        self.depth = parent.depth
        self.children = []
        self.prob = action_and_score[1]  # the probability that this action should be token, provided by default policy
        self.sampled_returns = []

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
