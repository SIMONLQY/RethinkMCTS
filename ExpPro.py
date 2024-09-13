# -*- coding:utf-8 _*-
import numpy as np
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('RethinkMCTS')] + 'RethinkMCTS')  # 这里要改为你自己的项目的主目录
import re

import os
import warnings
from utils import get_proj_path, get_raw_data_path
import json
from tqdm import tqdm

# 忽略warning的输出
warnings.filterwarnings('ignore')


def process_results(results, save=None, filter_indices=None):
    """
    Print the results and log, also return it possibly for analysis
    """
    rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
        results['rewards'], results['rewards_train'], results['times'], results['sample_times'], results['compile_errors'], results['runtime_errors']
    if filter_indices is not None:
        filter_op = lambda result: {k: result[k] for k in result.keys() if k in filter_indices}
    else:
        filter_op = lambda result: result
    # ignore the keys and get the values, and convert to np.array
    convert_to_array = lambda result: np.array(list(result.values()))

    # filter all dictionaries by filter_indices, and convert them to numpy array
    rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
        map(lambda x: convert_to_array(filter_op(x)), [rewards, rewards_train, times, sample_times, compile_errors, runtime_errors])

    print(f"Got {len(times)} programs, averaging over {len(rewards)} programs")

    stat = {
        'result_dir': save,
        'pass_rate': 100 * np.mean(rewards),
        'ac': 100 * np.mean(np.array(rewards) == 1.),
        'training_pass_rate': 100 * np.mean(rewards_train) if None not in rewards_train else None,
        'training_ac': 100 * np.mean(rewards_train == 1.) if None not in rewards_train else None,
        'time': np.mean(times),
        'sample_times': np.mean(sample_times),
        'avg_compile_errors': 100 * np.mean(compile_errors),
        'has_compile_errors': 100 * np.mean(compile_errors > 0),
        'avg_runtime_errors': 100 * np.mean(runtime_errors),
        'has_runtime_errors': 100 * np.mean(runtime_errors > 0),
    }
    print("On the test set")
    print(f"Test Case Average (average accuracy over problems) = {stat['pass_rate']} %")
    print(f"Strict Accuracy (all test cases passed / total problems) = {stat['ac']} %")
    print()

    print("On the training set")
    print(f"Test Case Average (average accuracy over problems) = {stat['training_pass_rate']} %")
    print(f"Strict Accuracy (all test cases passed / total problems) = {stat['training_ac']} %")
    print()

    print('avg compile errors', stat['avg_compile_errors'])
    print('has compile errors', stat['has_compile_errors'])
    print('avg runtime errors', stat['avg_runtime_errors'])
    print('has runtime errors', stat['has_runtime_errors'])

    if len(times) > 0:
        print(f"Computation time (sec.): Mean: {stat['time']}, SE: {np.std(times) / np.sqrt(len(times))}, "
              f"Max: {np.max(times)}, Min: {np.min(times)}")
    else:
        print("No computation time recorded, or none was successfully run.")

    if len(sample_times) > 0:
        print(f"Sample times: Mean: {stat['sample_times']}, SE: {np.std(sample_times) / np.sqrt(len(sample_times))}, "
              f"Max: {np.max(sample_times)}, Min: {np.min(sample_times)}")
    else:
        print("No sample times recorded.")

    return stat


def eval_and_save_problems(args, indices):
    """
    Args:
        args: command arguments
        indices: the indices of problems to be evaluated
    """
    all_results_loc = os.path.join(args.save, f"all_results.json")
    # try:
    #     with open(all_results_loc, 'r') as f:
    #         all_results = json.load(f)
    #
    #     rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
    #         all_results['rewards'], all_results['rewards_train'], all_results['times'], all_results['sample_times'], all_results['compile_errors'], all_results['runtime_errors']
    # except:
    #     print(f"{all_results_loc} specified, but failed to open.")
    rewards, rewards_train, times, sample_times, compile_errors, runtime_errors, codes = {}, {}, {}, {}, {}, {}, {}
    input_token_nums = {}
    output_token_nums = {}

    # don't show progress bar if only one problem to test
    indices = tqdm(indices) if len(indices) > 1 else indices

    total_code_num_train = 0.0
    total_code_num_test = 0.0
    train_success_code_num = 0.0
    final_get_train_suc_num = 0.0
    test_success_code_num = 0.0
    total_train_reward = 0.0

    train_test_trans_fail_num = 0.0

    for index in indices:
        # if index >= 4088:  # 只看前xx道题目
        #     continue
        if str(index) in rewards.keys() and not args.retest:
            # print(f"skipping {index} because it's already in results json file")
            continue

        code_loc = os.path.join(args.save, f"{index}.json")

        if not os.path.exists(code_loc):
            # print(f"didn't find result for {index}")
            continue
        else:
            # result_loc exists, simply read it
            try:
                with open(code_loc) as f:
                    result = json.load(f)

                    reward, reward_train, time, sample_time = result['rewards'], result['train rewards'], result['time'], result['sample times']
                    if 'input_token_num' in result.keys():
                        input_token_nums[index] = int(result['input_token_num'])
                    if 'output_token_num' in result.keys():
                        output_token_nums[index] = int(result['output_token_num'])
            except Exception as e:
                print(f"failed to read {code_loc}, {e}")
                continue

        if len(reward) == 0 or (isinstance(time, list) and len(time) == 0):
            print(f"{code_loc} results non-parsable.")
            continue

        for j, train_score in enumerate(reward_train):
            if train_score == 1.0 and reward[j] != 0.0:
                train_test_trans_fail_num += 1

        if len(reward_train) > 0:
            total_code_num_train += len(reward_train)
            total_code_num_test += len(reward)
            train_success_code_num += reward_train.count(1.0)
            if train_success_code_num > 0:
                final_get_train_suc_num += 1
            test_success_code_num += reward.count(1.0)
            total_train_reward += sum(reward_train)

            # sort the training rewards of the samples, get the top n of them
            top_n_indices = np.argsort(reward_train)[::-1][:args.n]  # arges.n = 1
            # find the one that has the highest test reward
            return_index = max(top_n_indices, key=lambda x: reward[x])
        else:
            return_index = 0

        # add to the list
        rewards[index] = reward[return_index]

        # get best-possible result
        # if 1.0 in reward:
        #     rewards[index] = 1.0
        # else:
        #     rewards[index] = reward[return_index]

        codes[index] = result['codes']

        rewards_train[index] = reward_train[return_index] if len(reward_train) > 0 else 0
        # these values are None for failed experiments
        if time is not None: times[index] = time

        sample_times[index] = sample_time

        try:
            compile_errors[index] = result['compile_error']
            runtime_errors[index] = result['runtime_error']
        except:
            compile_errors[index] = 0
            runtime_errors[index] = 0

    print(f"avg train reward: {total_train_reward/total_code_num_train}")
    print(f"train success rate: {train_success_code_num} / {total_code_num_train}={train_success_code_num / total_code_num_train}")
    print(f"final train success rate: {final_get_train_suc_num} / {len(indices)}={final_get_train_suc_num / float(len(indices))}")
    print(f"test success rate: {test_success_code_num} / {total_code_num_test}={test_success_code_num / total_code_num_test}")
    print(f"train test transfer failed ratio: {train_test_trans_fail_num}/{total_code_num_test}={train_test_trans_fail_num/total_code_num_test}" )
    # save results to file
    all_results = {
        'rewards': rewards,
        'rewards_train': rewards_train,
        'times': times,
        'sample_times': sample_times,
        'compile_errors': compile_errors,
        'runtime_errors': runtime_errors,
        'codes': codes,
        'input_token_nums': input_token_nums,
        'output_token_nums': output_token_nums,
    }

    with open(all_results_loc, "w") as f:
        try:
            json.dump(all_results, f)
        except Exception as e:
            print(f"Couldn't save all results.\n{e}")

    # return results from args.start to args.end
    filter_op = lambda x: {k: x[k] for k in x.keys() if int(k) in indices}
    ret_results = {k: filter_op(v) for k, v in all_results.items()}

    return ret_results


def get_exp_results(exp_dir_name, exp_labels, playlist, disjoint_flag=False):
    import argparse

    parser = argparse.ArgumentParser(description="Get experiments results")
    parser.add_argument("--save", type=str, default="./results/", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("-n", default=1, type=int, help='Evaluate using the n best program candidates (the n programs that have the highest pass rate on the training set.')
    parser.add_argument('--retest', action='store_true', default=False, help="rerun tests.")

    args = parser.parse_args()
    unsolved_problems = {}
    train_rewards = {}
    test_rewards = {}
    all_results = []
    all_exp_solved = []

    for exp_id in playlist:
        args.save = f"{exp_dir_name}/Experiment_{exp_id}"
        if not os.path.exists(args.save):
            raise ValueError(f"Directory {args.save} does not exist.")
        files = os.listdir(args.save)
        indices = [int(re.findall(r'\d+', file)[0]) for file in files if (file.endswith('.json') and 'result' not in file)]

        # 只看前100道题目
        # tmp_indices = [int(re.findall(r'\d+', file)[0]) for file in files if (file.endswith('.json') and 'result' not in file)]
        # indices = []
        # for id in tmp_indices:
        #     if id < 4080:
        #         indices.append(id)

        results = eval_and_save_problems(args, indices)
        all_results.append(results)

        # 打印unsolved problem信息
        print('----------------------------------')
        print(f'Experiment {exp_id} unsolved problems:')
        for index in results['rewards'].keys():
            # 打印错误题目
            # if True:
            # if index == 115:
            if results['rewards'][index] != 1.0 and len(playlist) == 1:
                print(f'\n----')
                print(f"reward: {results['rewards'][index]}")
                print(f"unsolved problem {index}")
                print(results['codes'][index])
            if results['rewards'][index] == 1.0 and len(playlist) == 1:
                print(f'\n----')
                print(f"reward: {results['rewards'][index]}")
                print(f"a-sovled problem {index}")
                print(results['codes'][index])
            if exp_id == playlist[0]:
                if results['rewards'][index] == 1.0:
                    all_exp_solved.append(index)
            if results['rewards'][index] != 1.0 and index in all_exp_solved:
                all_exp_solved.remove(index)



        if disjoint_flag:
            unsolved_problems[exp_id] = []
            train_rewards[exp_id] = {}
            test_rewards[exp_id] = {}
            if len(playlist) != 2:
                print('disjoint_flag is only valid for two experiments')
                disjoint_flag = False
            for index in results['rewards'].keys():
                if results['rewards'][index] != 1.0:
                    unsolved_problems[exp_id].append(index)
                train_rewards[exp_id][index] = results['rewards_train'][index]
                test_rewards[exp_id][index] = results['rewards'][index]
        # stat = process_results(results, args.save)
        # stats[k] = stat

        """
        Print the results and log, also return it possibly for analysis
        """
        rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
            results['rewards'], results['rewards_train'], results['times'], results['sample_times'], results['compile_errors'], results['runtime_errors']
        filter_op = lambda result: result
        # ignore the keys and get the values, and convert to np.array
        convert_to_array = lambda result: np.array(list(result.values()))

        # filter all dictionaries by filter_indices, and convert them to numpy array
        rewards, rewards_train, times, sample_times, compile_errors, runtime_errors = \
            map(lambda x: convert_to_array(filter_op(x)), [rewards, rewards_train, times, sample_times, compile_errors, runtime_errors])

        print(f"Got {len(times)} programs, averaging over {len(rewards)} programs")

        stat = {
            'result_dir': args.save,
            'pass_rate': 100 * np.mean(rewards),
            'ac': 100 * np.mean(np.array(rewards) == 1.),
            'training_pass_rate': 100 * np.mean(rewards_train) if None not in rewards_train else None,
            'training_ac': 100 * np.mean(rewards_train == 1.) if None not in rewards_train else None,
            'time': np.mean(times),
            'sample_times': np.mean(sample_times),
            'avg_compile_errors': 100 * np.mean(compile_errors),
            'has_compile_errors': 100 * np.mean(compile_errors > 0),
            'avg_runtime_errors': 100 * np.mean(runtime_errors),
            'has_runtime_errors': 100 * np.mean(runtime_errors > 0),
        }
        print(f"{exp_labels[exp_id]}: PassRate: {stat['pass_rate']:.4f}%; AC: {stat['ac']:.4f}%; AVGTime: {stat['time']:.2f}s; Input token num: {np.sum(list(results['input_token_nums'].values()))}; Output token num: {np.sum(list(results['output_token_nums'].values()))}")
    print(f"all_exp_solved: {all_exp_solved}")
    if disjoint_flag:

        print('----------------------------------')
        more_unsolved_0 = []
        for id in unsolved_problems[playlist[0]]:
            if id not in unsolved_problems[playlist[1]]:
                more_unsolved_0.append(id)
                print(f"----{id}-1(wrong)")
                print(f"{all_results[0]['codes'][id]}")
                print(f"----{id}-2(correct)")
                print(f"{all_results[1]['codes'][id]}")
        more_unsolved_1 = []
        for id in unsolved_problems[playlist[1]]:
            if id not in unsolved_problems[playlist[0]]:
                more_unsolved_1.append(id)
        # print(f"{exp_labels[playlist[0]]} all unsolved problems: {unsolved_problems[playlist[0]]}")
        # print(f"{exp_labels[playlist[1]]} all unsolved problems: {unsolved_problems[playlist[1]]}")
        print(f"{exp_labels[playlist[0]]} more unsolved problems: {more_unsolved_0}")
        print(f"{exp_labels[playlist[1]]} more unsolved problems: {more_unsolved_1}")


        ############
        tmp_sta = {'train_big': 0,
                   'train_small': 0,
                   'test_big': 0,
                   'test_small': 0,
                   'train_big_while_test_small': 0,
                   'train_equal_while_test_small': 0,
                   'train_big_while_test_big':0,
                   'train_equal_while_test_big':0,
                   'train_small_while_test_small_ques':[]}
        for id in train_rewards[playlist[0]].keys():
            first_train_reward = train_rewards[playlist[0]][id]
            first_test_reward = test_rewards[playlist[0]][id]
            second_train_reward = train_rewards[playlist[1]][id]
            second_test_reward = test_rewards[playlist[1]][id]
            # print(f"{first_train_reward} - {second_train_reward} ({playlist[0]}-{playlist[1]}) {first_test_reward} - {second_test_reward}")
            if first_train_reward > second_train_reward:
                tmp_sta['train_big'] += 1
                # print(f"{first_train_reward}-{second_train_reward}")
                if first_test_reward < second_test_reward:
                    tmp_sta['train_big_while_test_small'] += 1
            if first_train_reward == second_train_reward and first_test_reward < second_test_reward:
                tmp_sta['train_equal_while_test_small'] += 1
                # print(id)
                # print(f"{first_train_reward} - {second_train_reward} ({playlist[0]}-{playlist[1]}) {first_test_reward} - {second_test_reward}")
            if first_train_reward > second_train_reward and first_test_reward > second_test_reward:
                tmp_sta['train_big_while_test_big'] += 1
            if first_train_reward == second_train_reward and first_test_reward > second_test_reward:
                tmp_sta['train_equal_while_test_big'] += 1
            if first_train_reward < second_train_reward:
                tmp_sta['train_small'] += 1

            if first_test_reward > second_test_reward:
                tmp_sta['test_big'] += 1
            if first_test_reward < second_test_reward:
                tmp_sta['test_small'] += 1

            if first_train_reward < second_train_reward and first_test_reward < second_test_reward:
                tmp_sta['train_small_while_test_small_ques'].append(id)

        print(tmp_sta)



if __name__ == "__main__":
    # APPS
    APPS_dir_name = f'{get_proj_path()}/results/apps/'
    APPS_labels = ['0 exp_0_name',
                   '1 exp_1_name',
                   '2 '
                   ]

    # Humaneval
    Humaneval_dir_name = f'{get_proj_path()}/results/humaneval/'
    Humaneval_labels = ['0 ',
                        ]

    dataset = 'apps'
    # dataset = 'humaneval'
    playlist = [2]  # 前面放新跑的实验，后面放旧的/要超过的baseline

    disjoint_flag = True

    if dataset == 'apps':
        ma_flag = True
        seperate = False
        exp_dir_name = APPS_dir_name
        exp_labels = APPS_labels

    elif dataset == 'humaneval':
        ma_flag = True
        seperate = False
        exp_dir_name = Humaneval_dir_name
        exp_labels = Humaneval_labels

    # get_exp_results
    get_exp_results(exp_dir_name, exp_labels, playlist, disjoint_flag)
