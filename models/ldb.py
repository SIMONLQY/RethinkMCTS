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


MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def trim_header(func_impl):
    if IMPORT_HEADER in func_impl:
        func_impl = func_impl.replace(IMPORT_HEADER, "")
    return func_impl

def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])

def print_messages(messages: List[Message], prefix = "") -> None:
    print("::CHAT MESSAGE::" +prefix)
    for msg in messages:
        print(msg.content)
    print("==================")


def divide(prog):
    try:
        cfg = CFGBuilder().build_from_src('block', prog)
    except Exception as e:
        return None, str(e)
    divided_block = []
    prog_lines = prog.split("\n")
    for block in cfg:
        if block.at() is None or block.end() is None:
            continue
        divided_block.append([block, prog_lines[block.at():block.end()+1], block.id])
    return divided_block, None

def get_trace_line(trace, funcname, fname):
    mark = f"--- modulename: .tmp.py, funcname: {funcname}" + "\n"
    lines = trace.split(mark)[1].split("\n")
    traces = []
    for l in lines:
        # trace also record comment lines for some reason
        if l.lstrip().startswith("\'\'\'") or l.lstrip().startswith("\"\"\"") or l.lstrip().startswith("#"):
            continue
        traces.append(l)
    return traces


def get_error_msg(error):
    error_lines = error.split('\n')
    error_msg = ""
    last_l = ""
    code = ""
    for l in error_lines:
        if "File \"" in last_l:
            code = l
        elif "Error: " in l:
            error_msg = ("This line is wrong: ```" + code + "```\n" + l) if "__var_list" not in code else l
            break
        last_l = l
    return error_msg


def get_trace(prog, funcname, test_input=None):
    fname = '.tmp.py.' + str(random.randint(0, 10000))
    f = open(fname, "w")
    f.write(prog)
    f.close()
    # run in command line python -m trace -t tmp.py > trace
    import subprocess
    try:
        if not test_input:
            res = subprocess.run(["python3", "-m", "trace", "-t", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        else:
            res = subprocess.run(["python3", "-m", "trace", "-t", fname], input=test_input.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except AssertionError:
        # This is expected if fail the test assetion
        pass
    except subprocess.TimeoutExpired:
        return "*timeout*"
    except Exception as e:
        error_msg = get_error_msg(res.stderr.decode('utf-8'))
        print("Trace Execution Fail:" + error_msg)
        return "*execution fail*" + error_msg
    finally:
        os.remove(fname)
    trace = res.stdout.decode('utf-8')
    # Find --- modulename: tmp, funcname: {funcname}
    try:
        trace = get_trace_line(trace, funcname, fname)
    except IndexError:
        ferr_name = "../error/.error.py" + str(time.time())
        return f"*parse fail*{ferr_name}"
    # Find all lines with .tmp.py
    line_trace = []
    for l in trace:
        if l.startswith(fname):
            import re
            m = re.search(f"^{fname}", l)
            if (not line_trace) or (line_trace[-1] not in l):
                line_trace.append(l[m.end():])
    return line_trace


def get_after(stmts):
    for s in stmts:
        if s == "":
            continue
        else:
            return s.strip(), int((len(s) - len(s.lstrip()))/4)

def get_range(prog, entry):
    tree = AstroidBuilder().string_build(prog)
    for ele in tree.body:
        if isinstance(ele, nodes.FunctionDef) and ele.name == entry:
            return [ele.lineno-1, ele.end_lineno-1] # Lineno start from 0
    return None

def get_lineno(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return int(trace_line[match.start()+1:match.end()-2])

def get_line(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    return trace_line[match.end()+1:]

def get_indent(trace_line):
    match = re.search("\([0-9]+\):", trace_line)
    len1 = len(trace_line[match.end()+1:])
    len2 = len(trace_line[match.end()+1:].lstrip())
    return int((len1-len2)/4)

def extract_value(output):
    output = output.split("\n")[:-1]
    output = [x for x in output if x.startswith('Value_')]
    return output


def instrument_simple_block(prog, entry, divided_blocks):
    stmts = prog.split("\n")
    # Get range of entry function
    rang = get_range(prog, entry)
    block_insert = set([b[0].at() - 1 for b in divided_blocks] + [b[0].end() for b in divided_blocks])
    if rang is None:
        assert False, f"{entry} not in {prog}!"
    res = []
    for i, stmt in enumerate(stmts):
        if i < rang[0]:
            res.append(stmt)
            continue
        elif i > rang[1]:
            res.append(stmt)
            break
        if (i+1) not in block_insert:
            res.append(stmt)
            continue
        # indent the same as this statement
        refs, indent_after = get_after(reversed(stmts[:i+1]))
        # Unless
        if refs.startswith("else:") or refs.startswith("elif ") or refs.startswith("if ") or refs.startswith("while ") or refs.startswith("for ") or refs.startswith("def "):
            refs, indent_after = get_after(stmts[i+1:])
        payload = "    "*indent_after + f"__var_list = vars();print(f'Value_After:{i+1}|' + '|'.join([(x + '=' + _str(__var_list[x])) for x in __var_list if not x.startswith('__')]));"
        if stmt.find(" return ") != -1:
            stmt = stmt.replace(" return ", " _ret = ")
            payload = payload + " return _ret"
        res.append(stmt)
        res.append(payload)
    return "\n".join(res)


def collect_runtime_value_simple(value_prof_prog, test_input=None):
    hook = ""
    import sys
    hooked_prog = hook + "\n" + value_prof_prog
    fname = "tmp_line.py" + f".{random.randint(0,10000)}"
    with open(fname, "w") as f:
        f.write(hooked_prog)
    import subprocess
    try:
        if not test_input:
            res = subprocess.run(["python3", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        else:
            res = subprocess.run(["python3", fname], input=test_input.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except subprocess.TimeoutExpired:
        return "*timeout*"
    finally:
        os.remove(fname)
    output = res.stderr.decode('utf-8')
    if "Traceback (most recent call last):" in output and ("AssertionError" not in output):
        output = get_error_msg(output)
        return "*execution fail*" + output
    output = res.stdout.decode('utf-8')
    return output

def parse_runtime_value_simple_block(output, trace_lines):
    trace_idx = 0
    blocks = []
    blk = []
    value_profiles = extract_value(output)
    trace_len = len(trace_lines)
    trace_linenos = [get_lineno(l) for l in trace_lines]
    last_bp = ""
    trace_idx = 0
    for i, l in enumerate(value_profiles):
        if trace_idx >= trace_len:
            break
        lineno = int(l.split(':')[1].split('|')[0])
        values = '\t'.join(l.split('|')[1:])
        values = values if len(values) < 100 else (values[:50] + "..." + values[-50:])
        if lineno not in trace_linenos:
            #payload = "    "*get_indent(trace_lines[trace_idx]) + "# " + values
            last_bp = values
            continue
        blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + last_bp)
        while trace_idx < trace_len and get_lineno(trace_lines[trace_idx]) != lineno:
            trace_l = trace_lines[trace_idx]
            blk.append(get_line(trace_l))
            trace_idx += 1
        if trace_idx == trace_len:
            break
        blk.append(get_line(trace_lines[trace_idx]))
        blk.append("    "*get_indent(trace_lines[trace_idx]) + "# " + values)
        last_bp = values
        blocks.append(blk)
        blk = []
        trace_idx += 1
    if trace_idx < trace_len:
        blk = ["    "*get_indent(trace_lines[trace_idx]) + "# " + last_bp] + blk
        while trace_idx < trace_len:
            blk.append(get_line(trace_lines[trace_idx]))
            trace_idx += 1
        blocks.append(blk)
    return blocks


def std_inout_package(prog):  # for apps standard input output problem
    tmp_prog = prog.split("\n")

    new_prog = []
    for x in tmp_prog:
        if (not x.startswith("from ")) and (not x.startswith("import ")):
            new_prog.append("    " + x + "\n")
        else:
            new_prog.append(x + "\n")
    tmp_prog = new_prog

    new_prog = ""
    started = False
    for i in tmp_prog:
        if i.startswith("    ") and not started:
            # new_prog += "stdin = sys.stdin\nstdout = sys.stdout\n"
            new_prog += "def code():\n"
            new_prog += i
            started = True
        elif started and ((i.startswith("from ")) or (i.startswith("import "))):
            new_prog += "    " + i
        else:
            new_prog += i
    tmp_prog = new_prog
    return tmp_prog

def get_code_traces_block(prog, test, entry):
    if entry == None:
        entry = "code"
        prog = std_inout_package(prog)

    log_of_tracing = ""
    # Divide program into basic block units
    divided_blocks, error = divide(prog)
    prog_lines = prog.split("\n")
    if divided_blocks is None:
        return "*execution fail*" + error
    # Collect Execution Traces
    if test.find("assert ") != -1:
        test = test.replace("assert ", "print(").split(" == ")[0] + ")"

    if entry != 'code':
        exec_prog = prog + "\n" + test
        trace_lines = get_trace(exec_prog, entry)
    else:
        exec_prog = prog + "\n" + 'code()'
        trace_lines = get_trace(exec_prog, entry, test_input=test)
    if isinstance(trace_lines, str):
        if trace_lines == "*timeout*" or trace_lines.startswith("*execution fail*") or trace_lines.startswith("*parse fail*"):
            return trace_lines
    log_of_tracing += str("Trace:\n"+ '\n'.join(trace_lines[:10]))
    value_prof_prog = instrument_simple_block(prog, entry, divided_blocks)
    log_of_tracing += str("\nValue Profile Program:\n" + value_prof_prog + "\n" + test + "\n")
    if entry != 'code':
        output = collect_runtime_value_simple(value_prof_prog + "\n" + test)
    else:
        output = collect_runtime_value_simple(value_prof_prog + "\n" + 'code()', test_input=test)
    if output == "*timeout*" or output.startswith("*execution fail*"):
        return output
    log_of_tracing += "\n" + str("Value Profile Output:\n" + output)
    runtime_value = parse_runtime_value_simple_block(output, trace_lines)
    if not os.path.exists("./tracing_log"):
        os.makedirs("./tracing_log")
    log_file = "./tracing_log/trace.log."+str(random.randint(0, 10000))
    with open(log_file, 'w') as f:
        f.write(log_of_tracing)
        print(f"Writing tracing logs to {log_file}")
    return runtime_value

def ldb_debug(model, messages, prompt: str, prev_func_impl: str, failed_test: str, entry: str,
              dataset_type: str = "", level: str = "block") -> str:
    if '__name__' in prev_func_impl and "__main__" in prev_func_impl:
        tmp_codes = prev_func_impl.split("\n")
        begin_main = False
        new_code = ""
        for line in tmp_codes:
            if begin_main:
                if line.startswith('\t'):
                    line = line[1:]
                elif line.startswith('    '):
                    line = line[4:]

            if not ('__name__' in line and '__main__' in line):
                new_code = new_code + line + "\n"
            else:
                begin_main = True

        prev_func_impl = new_code
        # print(code)
    if "```python" in prev_func_impl:
        prev_func_impl = prev_func_impl.split("```python")[1].split("```")[0]
    prev_func_impl = trim_header(prev_func_impl)
    if "# Current Execution Output:" not in failed_test:
        if failed_test is None:
            failed_test = 'None'
        with open("error.log", "a") as f:
            f.write(f"\n\n {failed_test}")
        assert 0
    if dataset_type == 'humaneval':
        failed_test_string = failed_test.split("# Current Execution Output:")[0]
        real_test_output = failed_test.split("# Current Execution Output:")[1]
    elif dataset_type == 'apps':
        failed_test_string = failed_test.split("# Current Execution Output:")[0]
        failed_test_string_input = failed_test_string.split("# Input:\n")[1].split("\n# Ground Truth Output:")[0]
        failed_test_string_output = failed_test_string.split("# Ground Truth Output:\n")[1].strip()
        real_test_output = failed_test.split("# Current Execution Output:")[1]
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported.")
    if model.is_chat:
        if dataset_type in ["humaneval", "MBPP", "apps"]:
            if len(messages) == 0:
                messages = [
                    Message(
                        role="system",
                        content="You are an expert programming assistant.",
                    ),
                    Message(
                        role="user",
                        content=f"Complete the following task in Python. Please respond with code only (with the code inside a Markdown code block).\n{prompt}"
                    ),
                    Message(
                        role="assistant",
                        content=f"{prev_func_impl}"
                    )
                ]
                print_messages(messages, "268:\n")
            feedback = f"The code above fails the given unit test:\n{failed_test}. \nHelp me debug this.\n"
        # Check whether the solution can be executed
        if dataset_type == 'humaneval':
            trace_blocks = get_code_traces_block(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
        elif dataset_type == 'apps':
            trace_blocks = get_code_traces_block(IMPORT_HEADER + prev_func_impl, failed_test_string_input, entry)

        print("Get trace blocks...")
        # CANNOT EXECUTED
        if isinstance(trace_blocks, str):
            if trace_blocks == "*timeout*":
                print("The program exceeds the time limit!")
                msg = [Message(role="user", content=f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            elif trace_blocks.startswith("*execution fail*"):
                print(trace_blocks.replace("*execution fail*", ""))
                msg = [Message(role="user", content=f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            elif trace_blocks.startswith("*parse fail*"):
                print("The program is weird")
                msg = [Message(role="user", content=f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            else:
                assert False, "Strange type of trace error: " + trace_blocks
            print_messages(msg)
            messages += msg
            return '', messages
        elif len(trace_blocks) == 0:
            print("No trace blocks found.")
            msg = [Message(role="user", content=f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            print_messages(msg)
            messages += msg
            return'', messages
        # Start debugging
        msg = [Message(
            role="user",
            content=feedback + "\nHere is the code execution trace block by block with the intermediate variable values. Please explain the execution FOR EACH BLOCK and answer whether this block is correct or not. If not, give an explanation on what is wrong. Please wrap your response into a JSON object that contains keys `block` with the name of each block, key `correct` with value False or True, and key `explanation` with an explanation on the bug. \nExample Answers:\n{\"block\": \"BLOCK-1\", \"correct\": \"True\", \"explanation\": \"The block initializes variable `a` and `b`.\"}\n{\"block\": \"BLOCK-2\", \"correct\": \"False\", \"explanation\": \"The block is incorrect because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.\"}"
        )]

        max_num_blocks = 10
        if len(trace_blocks) > max_num_blocks:
            print("Sample trace block...")
            selected_blocks = trace_blocks[:int(max_num_blocks / 2)] + trace_blocks[-int(max_num_blocks / 2):]
            trace_blocks = selected_blocks
        trace_block_texts = ''
        for i, b in enumerate(trace_blocks):
            b = "\n".join(b)
            b = f"\n[BLOCK-{i}]\n" + b
            if len(model.tokenizer.encode(b, allowed_special={'<|endoftext|>'})) > 5000:
                tmp_shorter = model.tokenizer.encode(b, allowed_special={'<|endoftext|>'})[:5000]
                b = model.tokenizer.decode(tmp_shorter)
            trace_block_texts += b
            msg[0].content += b
        msg[0].content += "\n"
        messages += msg
        print_messages(msg)
        explanation_all = model.generate_chat(messages=messages, num_comps=1, temperature=0, stop=['[debug end]', 'Here is the updated code:'])

        # wrong_block, explanation = parse_explanation(explanation_all, trace_blocks, prev_func_impl)
        msg = [
            Message(
                role="assistant",
                content=explanation_all
            )
        ]
        print_messages(msg)
        messages += msg
    return trace_block_texts, messages


PY_CHAINOFDEBUG_TEXT2CODE_INSTRUCTION="""# Write Python function to complete the task and pass the assertion tests.
### Task Start ###
# These are the assertions for your function:
assert find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']

def find_char_long(text):
    \"\"\" Write a function to find all words which are at least 4 characters long in a string by using regex. \"\"\"
    if text == \"\":
        return []
    pat = r\"\\b\\w{4}\\b\"
    res = re.findall(pat, text)
    return res

Feedback: With the above function, the assertion is `find_char_long('Please move back to stream') == ['Please', 'move', 'back', 'stream']` but the real execution output is `['move', 'back']`.
Debug the program trace block by block until find the incorrect block. Every block should have different feedbacks:
[BLOCK-1]
    # text=\"Please move back to stream\"
    if text == \"\":
[BLOCK-2]
    # text="Please move back to stream"
    pat = r\"\\b\\w{4}\\b\"
    res = re.findall(pat, text)
    # text=\"Please move back to stream\" pat=\"\\b\\w{4}\\b\"  res=['move', 'back']
[debug]
[BLOCK-1]
Feedback: CORRECT. This block is correct. It checks if the input text is empty. If the input text is empty, it returns an empty list without do regex match.
[BLOCK-2]
Feedback: INCORRECT. This block defines a regular expression pattern `pat` with value r\"\\b\\w{4}\\b\". However, there's an issue with the regular expression pattern. It only matches words that are exactly 4 characters long. Therefore, the return value `_ret` is `['move', 'back']`. In the task description, it asks for words *which are at least 4 characters long*. To fix the code, we should change the line `pat = r\"\\b\\w{4}\\b\"` into `pat = r\"\\b\\w{4,}\\b\"`.
[/debug]
Please fix the Python code.
[python]
import re
def find_char_long(text):
    \"\"\" Write a function to find all words which are at least 4 characters long in a string by using regex. \"\"\"
    if text == \"\":
        return []
    pat = r\"\\b\\w{4,}\\b\"
    res = re.findall(pat, text)
    return res
[/python]
### Task End ###

### Task Start ###
# These are the assertions for your function:"""


class LDB:
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

        self.seed_codes = {}
        if self.args.arch == 'gpt3.5' and self.args.dataset == 'humaneval':

            seed_dir = f"{get_proj_path()}/dataProcess/{self.args.dataset}/seed/gpt-3.5-turbo-0125/"
            seed_path = os.path.join(seed_dir, "seed.jsonl")
            with jsonlines.open(seed_path) as reader:
                for item in reader:
                    idx = int(item['task_id'].split('/')[1])
                    self.seed_codes[idx] = item['solution']

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

            if len(self.seed_codes) == 0:
                code_id = self.generator.get_rationale_predicted_sequence(initial_state)
                cur_func_impl = self.tokenizer.decode(code_id)
            else:
                idx = int(problem_instance['task_id'].split('/')[1])
                cur_func_impl = '\n' + self.seed_codes[idx]
                code_id = self.tokenizer.encode(cur_func_impl)

            full_result = self.get_reward(code_id, with_verbal=True)
            complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

            if complete_prog_score >= 1:
                is_pass_train = True
                train_reward = complete_prog_score
                test_reward = self.get_reward(code_id, mode='test')
                break
            # use debug to iteratively improve
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
                        failed_test_list.append(verbal_feedback['output'])
                        tmp_count += 1
            messages = []
            max_iters = 10
            while cur_iter < max_iters:
                print('---------------')
                print('cur_iter:', cur_iter)
                print('---------------')

                cur_failed_test = failed_test_list[random.randint(0, len(failed_test_list) - 1)] if len(failed_test_list) >= 1 else None

                if self.args.dataset == 'humaneval':
                    trace_block_texts, messages = ldb_debug(self.generator, messages, self.tokenizer.decode(initial_state), cur_func_impl, cur_failed_test, entry=self.cur_prob_instance['entry_point'], dataset_type=self.args.dataset)
                elif self.args.dataset == 'apps':
                    trace_block_texts, messages = ldb_debug(self.generator, messages, self.tokenizer.decode(initial_state), cur_func_impl, cur_failed_test, entry=self.cur_prob_instance['method_name'], dataset_type=self.args.dataset)
                else:
                    raise ValueError("Unknown dataset type")

                # ldb_generate
                msg = [
                    Message(
                            role="user",
                            content=f"Please fix the Python code."
                        )
                ]
                messages += msg
                print_messages(msg)
                func_bodies = self.generator.generate_chat(messages=messages, num_comps=1, temperature=0, stop=['[debug end]', 'Here is the updated code:'])
                tmp_msg = [
                    Message(
                        role='assistant',
                        content=func_bodies
                    )
                ]
                messages += tmp_msg
                print_messages(tmp_msg)

                cur_func_impl = func_bodies
                code_id = self.tokenizer.encode(cur_func_impl)
                full_result = self.get_reward(code_id, with_verbal=True)
                complete_prog_score, verbal_feedbacks = full_result[0], full_result[1]

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
        output_dict['train_reward'] = train_reward
        output_dict['test_reward'] = test_reward
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
            with open("error.log", "a") as f:
                f.write(f"test framework exception = {repr(e)}{e}\n")
            if with_verbal:
                feedback_dict = {
                    'error': '',
                    'output': f"# Input:\nUnknown\n# Ground Truth Output:\n:\nUnknown\n\n# Current Execution Output: \nThe code executes failed! No result got!"
                }
                curr_res = [[False, feedback_dict]]
                verbal_feedbacks = [feedback_dict]
            else:
                curr_res = [-1]

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

    def convert_state_to_program(self, s):
        s = self.tokenizer.decode(s)
        if "ANSWER:" in s:
            s = s.split("ANSWER:\n")[1]
        s = s.replace("<|endoftext|>", "")
        return s
