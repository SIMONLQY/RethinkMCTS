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
import multiprocessing
import signal
from pyext import RuntimeModule
import sys
import faulthandler
from unittest.mock import patch, mock_open
from io import StringIO
import contextlib
import io
import platform
import traceback
import contextlib
import io
import sys
import ast
import astunparse

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
timeout = 10


class HumanevalExecutor:
    def __init__(self, args):
        self.args = args

    def check_correctness(self, prob_instance, code, mode, with_verbal=False):
        """
        Check if the code is correct
        """
        if not self.args.debug:
            manager = multiprocessing.Manager()
            result = manager.list()
            p = multiprocessing.Process(target=self._temp_run, args=(prob_instance, code, mode, result, with_verbal))
            p.start()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        else:
            # debugß
            result = []
            self._temp_run(prob_instance, code, mode, result, with_verbal)

        # result = []
        # self._temp_run(prob_instance, code, mode, result, with_verbal)

        if not result:
            # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
            # so we use 21=the average number of tests for a sample in the test split instead
            avg_number_tests = 21
            if not with_verbal:
                result = [[-1] * avg_number_tests]
            else:
                feedback_dict = {
                    'error': '',
                    'output': f"Unknown test\n\n# Current Execution Output: \nThe code executes failed! No result got!"
                }
                result = [[[False, feedback_dict]] * avg_number_tests]
        return result[0]

    def _temp_run(self, prob_instance, code, mode, result, with_verbal=False):
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]

        result.append(self.run_test(prob_instance=prob_instance, test=code, mode=mode, with_verbal=with_verbal))

    def run_test(self, prob_instance, test, mode, with_verbal=False):
        reliability_guard

        code = test

        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"

        if mode == 'train':
            # for_test_code = prob["example_test"]
            results = []
            for test_line in prob_instance['given_tests']:
                func_code = test_setup + code + "\n"
                assert_statement = '\n' + test_line
                results.append(self.execute_test(prob_instance, func_code, assert_statement, with_verbal))

            print('\n-------------------train')
            print(results)
            return results
        elif mode == 'generated_tests':
            # for_test_code = prob["example_test"]
            results = []
            for test_line in prob_instance['generated_tests']:
                func_code = test_setup + code + "\n"
                assert_statement = '\n' + test_line
                results.append(self.execute_test(prob_instance, func_code, assert_statement, with_verbal))

            print('\n-------------------generated tests')
            print(results)
            return results
        else:
            for_test_code = prob_instance["test"]

            results = []
            func_code = test_setup + code + "\n"
            assert_statement = '\n' + for_test_code
            results.append(self.execute_test(prob_instance, func_code, assert_statement, with_verbal))

            print('\n-------------------test')
            print(results)
            return results

    def execute_test(self, prob, func_call, assert_statement, with_verbal=False):
        def unsafe_execute():
            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir
            # Disable functionalities that can make destructive changes to the test.
            # reliability_guard()
            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        # WARNING
                        # This program exists to execute untrusted model-generated code. Although
                        # it is highly unlikely that model-generated code will do something overtly
                        # malicious in response to this test suite, model-generated code may act
                        # destructively due to a lack of model capability or alignment.
                        # Users are strongly encouraged to sandbox this evaluation suite so that it
                        # does not perform destructive actions on their host or network.
                        # Once you have read this disclaimer and taken appropriate precautions,
                        # uncomment the following line and proceed at your own risk:
                        test_code = func_call + '\n' + assert_statement
                        exec(test_code, exec_globals)
                    result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except AssertionError as e:
                result.append(f"failed: AssertionError")
            except BaseException as e:
                result.append(f"failed: {e}")
            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

        if not with_verbal:
            result = []
            unsafe_execute()
            if not result:
                result.append("timed out")
            return result[0] == "passed"
        else:
            """Execute the test code and return the error message"""
            comb_code = func_call + '\n' + assert_statement
            entry = prob['entry_point']
            error = exec_code(entry, comb_code)

            if error != '':
                if error == "TIMEOUT":
                    return [False, {'error': 'TIMEOUT', 'output': f"\n{assert_statement.strip()}\n\n# Current Execution Output: \nTIMEOUT"}]
                output = eval_code(entry, func_call, assert_statement)

                if output == '':
                    output = error

                failed_test = f"\n{assert_statement.strip()}\n\n# Current Execution Output: \n{output}"

                feedback_dict = {
                    'error': error,
                    'output': failed_test
                }
                return [False, feedback_dict]
            else:
                return [True, ""]


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left  # type: ignore
    except:
        call_str = ast_parsed.body[0].test  # type: ignore

    return astunparse.unparse(call_str).strip()


def exec_code(entry: str, code: str) -> str:
    """Execute the test code and return the error message"""
    comb_code = code

    error = ""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with time_limit(timeout):
                complied_comb_code = compile(comb_code, f"{entry}.py", "exec")
                exec(complied_comb_code, {"__builtins__": __builtins__})
    except TimeoutException:
        error = "TIMEOUT"
    except Exception:
        try:
            exc_type, exc_value, exc_tb = sys.exc_info()
            exc_info = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()

            tb_list = traceback.extract_tb(exc_tb)
            n_tb = 0
            while n_tb < len(tb_list) and tb_list[n_tb][0] != f"{entry}.py":
                n_tb += 1
            tb_list = tb_list[n_tb:]
            if len(tb_list) > 0:
                error_info_list = traceback.format_list(tb_list)
                comb_code_list = comb_code.split("\n")
                error_info_list = [
                    info_line.strip() + "\n" + comb_code_list[tb[1] - 1]
                    for (info_line, tb) in zip(error_info_list, tb_list)
                ]

                error_info = "\n".join(error_info_list).strip()
                error = f"{error_info}\n{exc_info}"
            else:
                error = f"{exc_info}"
        except Exception:
            error = "Unknown Error"
    return error


def eval_code(entry, func_call, assert_statement):
    """Eval the statement the given context return the result"""
    comb_code = func_call + '\n'

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            complied_comb_code = compile(comb_code, f"{entry}.py", "exec")
            _globals = {"__builtins__": __builtins__}
            func_call = get_call_str(assert_statement)

            exec(complied_comb_code, _globals)

            # result = eval(func_call, _globals)
            #
            # exec(f"from typing import *\n{func_call}", globals())
            result = function_with_timeout(eval, (func_call, _globals), timeout)
    except TimeoutException:
        result = "TIMEOUT"
    except Exception:
        result = ""
    return result


from threading import Thread


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        if self.is_alive():
            return None
        return self.ret

    def terminate(self):
        self._stop()


def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        thread.terminate()
        raise TimeoutError()
    else:
        return result_container[0]


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split("\n"))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def reliability_guard(maximum_memory_bytes=None):
    """
    source: https://github.com/openai/human-eval
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


class TimeoutException(Exception):
    pass
