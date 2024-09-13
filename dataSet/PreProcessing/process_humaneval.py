# -*- coding:utf-8 _*-
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('RethinkMCTS')] + 'RethinkMCTS')  # 这里要改为你自己的项目的主目录
import gzip
import json
from utils import *
from typing import *
import jsonlines


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def read_humaneval_dataset(
        data_file: str = None,
        dataset_type: str = "humaneval",
        num_shot=None,
) -> Dict:
    if num_shot is not None:
        print(f"{num_shot}-shot setting...")
    if "humaneval" in dataset_type.lower():
        if data_file is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_path, "..", "humaneval-x", "python", "data", "humaneval_python.jsonl.gz")
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset


def main():
    # 文件1
    # problem_file = f"{get_raw_data_path()}/humaneval-x/python/data/humaneval_python.jsonl.gz"
    problem_file = f"{get_proj_path()}/dataProcess/humaneval/processed.jsonl"
    problems = read_humaneval_dataset(problem_file, dataset_type="humaneval")  # 读取问题集，返回的是一个字典，key是task_id，value是一个字典，包括prompt, test, canonical_solution等

    for problem in problems.keys():
        print(problem)
        for key in problems[problem].keys():
            print('----------------')
            print(key)
            print(problems[problem][key])

        break
    assert 0

    # 文件2
    # with_given_tests_file = f"{get_proj_path()}/dataProcess/humaneval/tests.jsonl"
    # items = {}
    # with jsonlines.open(with_given_tests_file) as reader:
    #     for item in reader:
    #         items['Python' + item['task_id'].split('HumanEval')[1]] = item
    #
    # for problem in problems.keys():
    #     if problem in items:
    #         problems[problem]['given_tests'] = items[problem]['given_tests']
    #         problems[problem]['entry_point'] = items[problem]['entry_point']
    #     else:
    #         assert False

    # 写入文件3
    target_file = f"{get_proj_path()}/dataProcess/humaneval/processed.jsonl"
    # 将每个dict转换为JSON格式的字符串并写入文件
    # with open(target_file, "w") as file:
    #     for problem in problems.keys():
    #         item = problems[problem]
    #         json_line = json.dumps(item)
    #         file.write(json_line + "\n")

    with jsonlines.open(target_file) as reader:
        for item in reader:
            print('\nalsdjflkasjdlfajsdlfjalskfjlaskdjfla')
            for key in item.keys():
                print('----------------')
                print(key)
                print(item[key])

            break


if __name__ == "__main__":
    main()
