# -*- coding:utf-8 _*-
# Process directory
import sys
import os

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('RethinkMCTS')] + 'RethinkMCTS')
import warnings

warnings.filterwarnings('ignore')
import shutil

# CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# openai keys
import openai

os.environ["TOKENIZERS_PARALLELISM"] = "false"
API_KEY = "xxx"  # lqy
openai.api_key = API_KEY

os.environ["OPENAI_API_KEY"] = API_KEY

# imports
import torch
from Processor import Processor
from argparse import ArgumentParser
from accelerate import Accelerator
from utils import *
import pprint


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def main():
    parser = ArgumentParser("RethinkMCTS")
    parser.add_argument('-eid', '--experiment-idx', type=int, default=0, help='Experiment id for one model')
    parser.add_argument('-d', '--dataset', type=str, default='humaneval', choices=['apps', 'humaneval'])
    parser.add_argument('-m', '--model', type=str, default='rethinkmcts',
                        choices=['mcts_line', 'mcts_token', 'bs', 'sample', 'ldb', 'mcts_thought', 'mcts_code',
                                 'rethinkmcts', 'ToT', 'LATS', 'RethinkMCTSNoVerb',
                                 'RAP', 'LDB', 'Reflexion'])
    parser.add_argument("--rollout", default=16, type=int, help="The maximum number of rollouts.")
    parser.add_argument("--arch", default="gpt4o-mini",
                        choices=["gpt3.5", "gpt3.5completion", 'gpt4', 'gpt4o-mini', 'gpt4o'])
    parser.add_argument('--mctsvalue', type=str, default='test',
                        choices=['test', 'gpteval', 'gptevalTC', 'verbalMemory', 'verbalMemoHistory'],
                        help='The value function to use for MCTS.')
    parser.add_argument("--loadArchDir", default=f"", type=str)
    parser.add_argument("--save", type=str, default="./results", help="Directory to save generated code.")
    parser.add_argument('--rerun', action='store_true', default=False,
                        help="If True, rerun if the output file already exists.")
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument("--small_part_run", type=str2bool, default=True,
                        help="If True, the questions that all methods pass by gpt3.5 could be avoid. Only for advanced methods.")
    parser.add_argument('--json_save_all', type=str2bool, default=False, help='If True, save all json files.')
    # -------------------------------------------------------------------------------------------- dataset
    parser.add_argument("-i", "--index", default=None,
                        type=int)
    parser.add_argument("--start", default=4000, type=int)
    parser.add_argument("--end", default=4100, type=int)
    # -------------------------------------------------------------------------------------------- APPS
    parser.add_argument("--APPDifficulty", default="introductory", choices=['introductory', 'interview', 'competition'],
                        help="The difficulty of the problems to solve.")
    # -------------------------------------------------------------------------------------------- mcts
    parser.add_argument("--width", default=3, type=int, help="The maximum number of children for any node.")
    parser.add_argument("--horizon", default=16000, type=int, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-beams", default=1, type=int, help="The number of beams for beam search.")
    parser.add_argument("--num-samples", default=1, type=int, help="The number of samples for Sampling + Filtering.")
    parser.add_argument("--max-sample-times", default=768, type=int,
                        help="The maximum number of Transformer generation function calls."
                             "Program stops when this number is reached (default to be 512 * 1.5 = 768).")
    parser.add_argument("--ucb-constant", default=4., type=float)
    parser.add_argument("--ucb-base", default=10., type=float)
    parser.add_argument("--uct-alg", default="var_p_uct", choices=["uct", "p_uct", "var_p_uct"],
                        help="The UCT algorithm to use."
                             "`uct` is the original UCT algorithm,"
                             "`p_uct` is the UCT algorithm with PUCT,"
                             "and `var_p_uct` is the UCT algorithm with variable PUCT.")
    parser.add_argument('--top-k-cache-steps', type=int, default=1024,
                        help="Number of forward steps to cache top k caches, default 1024 means the whole horizon.")
    parser.add_argument("--public-cases-type", type=str, default='half',
                        help="Number of public test cases to use for evaluation.")
    parser.add_argument("--ts-mode", default="best", choices=["best", "sample"],
                        help="Tree search mode within the evaluation step. `best` uses beam search, `sample` uses sampling.")
    parser.add_argument("--max_think_times", default=2, type=int, help="The max num of think times")
    # --------------------------------------------------------------------------------------------
    parser.add_argument('--total_input_token_num', type=int, default=0, help='The maximum number of tokens to input.')
    parser.add_argument('--total_output_token_num', type=int, default=0, help='The maximum number of tokens to output.')
    parser.add_argument('--failed_json_num', type=int, default=0, help='The number of failed json format output.')
    parser.add_argument('--all_json_num', type=int, default=0, help='The number of all json format output.')
    parser.add_argument('--verbal_length_exd_num', type=int, default=0,
                        help='The number of length of verbal length too long')
    parser.add_argument('--verbal_length_check_num', type=int, default=0,
                        help='The number of length check of verbal feedback')
    parser.add_argument('--rollout_count', type=int, default=-1, help='The rollout count')
    parser.add_argument('--generate_tests_total', type=int, default=0, help='The generate tests count')
    parser.add_argument('--failed_generate_tests_count', type=int, default=0,
                        help='The failed generated tests count total')
    parser.add_argument('--rethink_total_nums', type=int, default=0, help='The total number of rethink')
    parser.add_argument('--rethink_effective_nums', type=int, default=0, help='The effective number of rethink')
    parser.add_argument('--rethink_failed_nums', type=int, default=0, help='The failed number of rethink')
    parser.add_argument('--rethink_success_nums', type=int, default=0, help='The success number of rethink')
    parser.add_argument('--no_rethink_success_num', type=int, default=0, help='The success number of no rethink')
    # --------------------------------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--cudaDevice', type=str, default='cuda')

    args = parser.parse_args()
    """
    APPS:
    intro: 4000-4999
    inter: 0000-2999
    competition: 3000-3999
    """
    if args.dataset == 'apps':
        if args.index is None:
            if args.APPDifficulty == 'introductory':
                args.start = 4000
                args.end = 4100
            elif args.APPDifficulty == 'interview':
                args.start = 0
                args.end = 100
            elif args.APPDifficulty == 'competition':
                args.start = 3000
                args.end = 3100

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if "gpt3.5" in args.arch or 'gpt4' in args.arch:
        args.loadArchDir = f"{args.arch}"
    else:
        raise ValueError('wrong arch!')

    if args.experiment_idx == 2:
        args.rerun = True

    args.save = f"{args.save}/{args.dataset}/Experiment_{args.experiment_idx}"
    print(f'save dir: {args.save}')

    if args.rerun:
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
            print(f"'{args.save}' has been removed and recreated.")

    os.makedirs(args.save, exist_ok=True)

    print(pprint.pformat(vars(args)))

    runner = Processor(args)
    runner.run()


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
