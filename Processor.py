# -*- coding:utf-8 _*-
from torch.utils.data import DataLoader, SequentialSampler
from utils import *
import time
from models import *
from torcheval.metrics import HitRate, ReciprocalRank
import torchmetrics
from dataSet import *
import json
import os
import shutil


class Processor(object):
    def __init__(self, args):
        self.args = args
        self.st = time.time()

        # 模型
        self.device = args.cudaDevice
        if args.model == 'mcts_token':
            self.model = MCTSToken(args)
        elif args.model == 'bs':
            self.model = BS(args)
        elif args.model == 'mcts_line':
            self.model = MCTSLine(args)
        elif args.model == 'mcts_thought':
            self.model = MCTSThought(args)
        elif args.model == 'mcts_code':
            self.model = MCTSCode(args)
        elif args.model == 'rethinkmcts':
            self.model = RethinkMCTS(args)
        elif args.model == 'ToT':
            self.model = ToT(args)
        elif args.model == 'LATS':
            self.model = LATS(args)
        elif args.model == 'RethinkMCTSNoVerb':
            self.model = RethinkMCTSNoVerb(args)
        elif args.model == 'RAP':
            self.model = RAP(args)
        elif args.model == 'LDB':
            self.model = LDB(args)
        elif args.model == 'Reflexion':
            self.model = Reflexion(args)

        # 数据
        self.problem_indices = []
        self.removed_list = []
        if args.index is not None:
            self.problem_indices = [args.index]
        elif args.end is not None:
            self.problem_indices = range(args.start, args.end)
        else:
            raise Exception("Don't know what problems to solve.")

        if args.dataset == 'apps':
            self.data_handler = APPSHandler(self.problem_indices, args)
        elif args.dataset == 'humaneval':
            if args.index is None:
                self.problem_indices = [i for i in range(164)]
            self.data_handler = HumanevalHandler(self.problem_indices, args)

    def run(self):
        for idx, problem_instance in zip(self.problem_indices, self.data_handler.problems):
            self.st = time.time()
            # 记录结果
            result_loc = os.path.join(self.args.save, f"{idx}.json")
            if not self.args.rerun:
                # if not forcing rerun, check if this experiment has run or failed before
                if os.path.exists(result_loc):
                    print(f"Found {result_loc}, args.rerun not enabled, skipping")
                    continue
            print(f"Solving Problem #{idx}")

            # 生成代码
            output_dict = self.model.generate(problem_instance)
            if output_dict is None:
                continue

            print(f"Final Program: \n{output_dict['final_program']}")
            print(f"train rewards: {output_dict['train_reward']}")
            print(f'test rewards: {output_dict["test_reward"]}')
            print(f'solve time: {time.time() - self.st}')
            print(f"Input token num: {self.args.total_input_token_num}, Output token num: {self.args.total_output_token_num}")
            Failed_json_rate = 0
            verbal_too_long_ratio = 0
            failed_generated_tests_ratio = 0
            rethink_effective_ratio = 0
            rethink_failed_ratio = 0
            rethink_success_ratio = 0
            if self.args.all_json_num != 0 and self.args.verbal_length_check_num != 0:
                Failed_json_rate = float(self.args.failed_json_num) / float(self.args.all_json_num)
                verbal_too_long_ratio = float(self.args.verbal_length_exd_num) / float(self.args.verbal_length_check_num)
                print(f"Failed json rate: {Failed_json_rate}")
                print(f"verbal_too_long_ratio: {verbal_too_long_ratio}")
            if self.args.generate_tests_total != 0:
                failed_generated_tests_ratio = float(self.args.failed_generate_tests_count) / float(self.args.generate_tests_total)
                print(f"failed_generated_tests_ratio: {failed_generated_tests_ratio}")
            if self.args.rethink_total_nums != 0:
                rethink_effective_ratio = float(self.args.rethink_effective_nums) / float(self.args.rethink_total_nums)
                rethink_failed_ratio = float(self.args.rethink_failed_nums) / float(self.args.rethink_total_nums)
                rethink_success_ratio = float(self.args.rethink_success_nums) / float(self.args.rethink_total_nums)
                print(f"rethink_effective_ratio: {rethink_effective_ratio}")
                print(f"rethink_failed_ratio: {rethink_failed_ratio}")
                print(f"rethink_success_ratio: {rethink_success_ratio}")
            with open(result_loc, "w") as f:
                json.dump({'codes': output_dict['final_program'],
                                'rewards': output_dict['all_test_rewards'],
                                'train rewards': output_dict['all_train_rewards'],
                                'time': time.time() - self.st,
                                'sample times': output_dict['avg_sample_time'],
                                'input_token_num': self.args.total_input_token_num,
                                'output_token_num': self.args.total_output_token_num,
                                'failed_json_rate': Failed_json_rate,
                                'failed_generated_tests_ratio': failed_generated_tests_ratio,
                                'verbal_too_long_ratio': verbal_too_long_ratio,
                                'rethink_effective_ratio': rethink_effective_ratio,
                                'rethink_failed_ratio': rethink_failed_ratio,
                                'rethink_success_ratio': rethink_success_ratio,
                                'rethink_total_nums': self.args.rethink_total_nums,
                                'rethink_failed_nums': self.args.rethink_failed_nums,
                                'rethink_success_nums': self.args.rethink_success_nums,
                                'rethink_effective_nums': self.args.rethink_effective_nums,
                                'no_rethink_success_num': self.args.no_rethink_success_num}, f)
                self.args.total_input_token_num = 0
                self.args.total_output_token_num = 0
