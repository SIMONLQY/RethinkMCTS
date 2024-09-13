# -*- coding:utf-8 _*-
import torch
from typing import List, Union, Optional, Literal
import dataclasses
import transformers
from utils import get_raw_data_path
from openai import OpenAI
import openai
import tiktoken
import time
from .cache import GPTTopKCache, GPTSeqCache
import math
import re
import json
import os
from time import sleep


def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.encode(l, allowed_special={'<|endoftext|>'}))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.encode(messages[0].content, allowed_special={'<|endoftext|>'}))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages


class GPTChat:
    def __init__(self, model_name, tokenizer, args, save_mid_json=[]):
        self.name = model_name
        self.is_chat = True
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.time_stamps = []
        self.ts_mode = args.ts_mode
        self.horizon = args.horizon
        self.client = OpenAI()
        self.terminal_token = self.tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
        self.width = args.width
        self.top_k_cache_steps = args.top_k_cache_steps
        self.top_k_cache = GPTTopKCache(args.width, cache_steps=args.top_k_cache_steps, tokenizer=tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.save_mid_json = save_mid_json

    def generate_chat(self, messages, stop, max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1):
        if self.args.arch == 'gpt3.5':
            model_name = 'gpt-3.5-turbo-0125'
        elif self.args.arch == 'gpt4':
            model_name = 'gpt-4-turbo-2024-04-09'
        elif self.args.arch == 'gpt4o-mini':
            model_name = 'gpt-4o-mini-2024-07-18'
        elif self.args.arch == 'gpt4o':
            model_name = 'gpt-4o'
        else:
            print(f'Model {self.args.arch} not implemented error!')
            assert 0
        for ti in range(20):  # 3次尝试机会
            sleep_interval = 7
            try:
                new_messages = change_messages(self.tokenizer, messages, 8000)
                messages = new_messages
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[dataclasses.asdict(message) for message in messages],
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps,
                    stop=stop
                )
            except Exception as e:
                print("GPT Error:", str(e))
                if "context_length_exceeded" in str(e):
                    messages = change_messages(self.tokenizer, messages, 8000)
                    print("AFTER CHANGE MESSAGE LEN:", len(messages))
                    print(messages)
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[dataclasses.asdict(message) for message in messages],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        n=num_comps,
                    )
                else:
                    sleep_t = sleep_interval * (ti + 1)
                    print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                    with open("error.log", "a") as f:
                        f.write(f"gpt failed multiple times with: {str(e)}\n")
                    sleep(sleep_t)
                    continue


            input_token_num = 0
            for msg in messages:
                input_token_num += len(self.tokenizer.encode(msg.content, allowed_special={'<|endoftext|>'}))
            output_token_num = len(self.tokenizer.encode(response.choices[0].message.content, allowed_special={'<|endoftext|>'}))
            self.args.total_input_token_num += input_token_num
            self.args.total_output_token_num += output_token_num

            if num_comps == 1:
                return response.choices[0].message.content  # type: ignore
            return [choice.message.content for choice in response.choices]  # type: ignore
        else:
            print(f'try failure with multiple times')
            assert False


    def generate_response_api(self, prompt, top_k, max_length=1024, system_message=None, temperature=0.0):
        sys_msg = "You are a helpful code generator that generate code to complete the given problem."
        if system_message:
            sys_msg = system_message
        for ti in range(20):
            sleep_interval = 7
            try:
                if self.args.arch == 'gpt3.5':
                    response = self.client.chat.completions.create(
                        # model='gpt-3.5-turbo-0613',
                        model='gpt-3.5-turbo-0125',
                        # model='gpt-3.5-turbo', # 最新的
                        # model='gpt-4',
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                        max_tokens=max_length,  # 调整生成文本的长度
                        temperature=temperature,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=True,
                        top_logprobs=top_k
                    )
                    message = response.choices[0].message.content
                    log_prob = response.choices[0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}
                elif self.args.arch == 'gpt4':
                    response = self.client.chat.completions.create(
                        model='gpt-4-turbo-2024-04-09',
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                        max_tokens=max_length,  # 调整生成文本的长度
                        temperature=temperature,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=True,
                        top_logprobs=top_k
                    )
                    message = response.choices[0].message.content
                    log_prob = response.choices[0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}
                elif self.args.arch == 'gpt4o':
                    response = self.client.chat.completions.create(
                        model='gpt-4o',
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                        max_tokens=max_length,  # 调整生成文本的长度
                        temperature=temperature,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    message = response.choices[0].message.content
                    log_prob = []
                elif self.args.arch == 'gpt4o-mini':
                    response = self.client.chat.completions.create(
                        model='gpt-4o-mini-2024-07-18',
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                        max_tokens=max_length,  # 调整生成文本的长度
                        temperature=temperature,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=True,
                        top_logprobs=top_k
                    )
                    message = response.choices[0].message.content
                    log_prob = response.choices[0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}
                elif self.args.arch == 'gpt3.5completion':
                    response = self.client.completions.create(
                        model="gpt-3.5-turbo-instruct",
                        prompt=prompt,
                        max_tokens=max_length,
                        temperature=0,
                        logprobs=top_k
                    )
                    message = response.choices[0].text
                    log_prob = response.choices[0].logprobs.top_logprobs

                input_token_num = len(self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'}))
                output_token_num = len(self.tokenizer.encode(message, allowed_special={'<|endoftext|>'}))
                self.args.total_input_token_num += input_token_num
                self.args.total_output_token_num += output_token_num
            except Exception as e:
                print("GPT Error:", str(e))
                if "context_length_exceeded" in str(e):
                    tmp_shorter = self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})[:4096]
                    prompt = self.tokenizer.decode(tmp_shorter)
                    if self.args.arch == 'gpt4o':
                        response = self.client.chat.completions.create(
                            model='gpt-4o',
                            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                            max_tokens=max_length,  # 调整生成文本的长度
                            temperature=temperature,
                            # top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                        )
                        message = response.choices[0].message.content
                        log_prob = []
                    else:
                        response = self.client.chat.completions.create(
                            # model='gpt-3.5-turbo-0613',
                            model='gpt-3.5-turbo-0125',
                            # model='gpt-3.5-turbo', # 最新的
                            # model='gpt-4',
                            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
                            max_tokens=max_length,  # 调整生成文本的长度
                            temperature=temperature,
                            # top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            logprobs=True,
                            top_logprobs=top_k
                        )
                        message = response.choices[0].message.content
                        log_prob = response.choices[0].logprobs.content  # 是一个length等于top k的list，每个位置是一个list{token: .., logprob:.., bytes:..}
                else:
                    sleep_t = sleep_interval * (ti + 1)
                    print(f"get {ti +1}, error: {e}, sleep {sleep_t} seconds")
                    with open("error.log", "a") as f:
                        f.write(f"gpt failed multiple times with: {str(e)}\n")
                    sleep(sleep_t)
                    continue
            return message, log_prob
        else:
            print(f'try failure with multiple times')
            assert False

    def generate_token_code_answer(self, input_ids, top_k=3, max_length=1024, max_new_tokens=None):
        input_prompt = self.tokenizer.decode(input_ids[0].tolist())

        with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                    f"Generate the code ONLY. No other explanation or words attached!\n") + input_prompt

        # print('\n-----------------1')
        # print(with_instru_input_prompt)

        if max_new_tokens:
            max_length = max_new_tokens
        response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k, max_length)

        # print('\n-----------------2')
        # print(response_text)

        sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})
        if self.args.arch == 'gpt3.5completion':  # completion 可能会出现的问题，sequence的encode会将['\n\n']识别成一个，但是实际上是一个一个换行符输出的
            if len(sequences) != len(log_probs):
                print('------------')
                print(sequences)
                tmp_list = []
                tmp_str = ''
                for tmp_dict in log_probs:
                    max_prob = -10000
                    max_prob_token = ' '
                    for key in tmp_dict.keys():
                        if tmp_dict[key] >= max_prob:
                            max_prob = tmp_dict[key]
                            max_prob_token = key
                    tmp_str = tmp_str + max_prob_token
                    tmp_list.append(self.tokenizer.encode(max_prob_token, allowed_special={'<|endoftext|>'})[0])
                sequences = tmp_list

        if self.args.arch == 'gpt3.5completion':
            sequences = input_ids[0].tolist() + sequences

        if len(log_probs) == 0:  # 之前已经生成了完整的程序,所以gpt判断不再需要token在后面
            log_probs = [{'<|endoftext|>': 1.0}]

        tmp_return = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0).to(self.device),
                                    scores=log_probs,
                                    attentions=None,
                                    hidden_states=None,
                                    beam_indices=None)

        return tmp_return

    def get_token_predict_sequence(self, state, horizon=None):
        """
        Args:
            horizon: return a new sequence with this extra length
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            encoded_ids = state  # as a list
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # use_seq_cache:
            output_ids = self.seq_cache.get(encoded_ids)
            if output_ids is not None:
                return output_ids

            model_output = self.generate_token_code_answer(
                input_ids,
                top_k=self.width,
                max_length=1024,
            )

            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(encoded_ids, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_token_predict(self, state):
        with torch.no_grad():
            if self.top_k_cache_steps > 0:
                top_k_info = self.top_k_cache.get(state)
                if top_k_info is not None:
                    return top_k_info

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            model_output = self.generate_token_code_answer(
                input_ids,
                top_k=self.width,
                max_new_tokens=1
            )

            if self.args.arch == 'gpt3.5completion':  # gpt3.5 completion
                top_scores = []
                top_tokens = []
                for top_token in model_output.scores[0].keys():
                    top_scores.append(math.exp(model_output.scores[0][top_token]))
                    top_tokens.append(self.tokenizer.encode(top_token, allowed_special={'<|endoftext|>'})[0])
                return top_tokens, top_scores
            elif self.args.arch in ['gpt3.5', 'gpt4', 'gpt4o-mini', 'gpt4o']:  # gpt3.5
                top_scores = []
                top_tokens = []
                for token_tops in model_output.scores:
                    top_scores.append([])
                    top_tokens.append([])
                    for token_probs in token_tops.top_logprobs:
                        top_scores[-1].append(math.exp(token_probs.logprob))
                        top_tokens[-1].append(self.tokenizer.encode(token_probs.token, allowed_special={'<|endoftext|>'})[0])
                return top_tokens[0], top_scores[0]
            else:
                raise ValueError('wrong arch!')

    def get_top_k_line_predict(self, state):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            with_instru_input_prompt = (f"Here is a problem to be solved by Python program. The program is now incomplete. \n"
                                        f"I need you to complete the program line by line, including line breaks and spaces. Remember you can only predict the next ONE line of the code, nothing else.\n"
                                        f"Here is the question and the incomplete program:\n") + input_prompt + f"\n The next line you predict is: "

            # print('\n-----------------1')
            # print(with_instru_input_prompt)

            top_scores = []
            top_lines = []
            top_lines_text = []

            for line_nums in range(self.width):
                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                if len(log_probs) == 0:  # 之前已经生成了完整的程序,所以gpt判断不再需要token在后面
                    log_probs = [{'<|endoftext|>': 1.0}]
                if "```python" in response_text:
                    response_text = response_text.split("```python")[1].split("```")[0]

                top_lines.append(self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'}))
                top_scores.append(1.0)
                top_lines_text.append(response_text)
                for positions in log_probs:
                    top_scores[-1] *= math.exp(positions.logprob)

            return top_lines, top_scores

    def get_top_k_line_predict_2(self, state):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            with_instru_input_prompt = (f"{input_prompt} \n "
                                        f"Here is a problem to be solved by Python program. The program is now incomplete. \n"
                                        f"I need you predict the next line of the program, including line breaks and spaces. "
                                        f"For the next adapting search algorithms, i need you to output {self.width} possible next lines. Remember each only contain the next ONE line of the code, nothing else.\n"
                                        f"Note that do not rush to solve the problem in this one line, generate the next line is ok.\n"
                                        f"Please wrap your response into a JSON object that contains keys `line` with the name of each line, and key `possibility` with the possibility of each line. \n"
                                        f"Example Answers:\n")
            with_instru_input_prompt += """
[
    {"line":"    print('Hello World')", "possibility": 0.9},
    {"line":"    print('Hello')", "possibility": 0.05},
    {"line":"    print('Hi')", "possibility": 0.05}
]
"""

            # print('\n-----------------1')
            # print(with_instru_input_prompt)

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=2048, temperature=0.0)
            # print('\n-----------------2')
            # print(response_text)

            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            try:
                response_text = json.loads(response_text)
                top_scores = []
                top_lines = []
                top_lines_text = []
                for ele in response_text:
                    top_scores.append(ele['possibility'])
                    ele['line'] = '\n' + ele['line']
                    top_lines.append(self.tokenizer.encode(ele['line'], allowed_special={'<|endoftext|>'}))
                    top_lines_text.append(ele['line'])
            except Exception as e:
                top_lines = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]

            return top_lines, top_scores

    def get_line_predict_sequence(self, state, horizon=None):
        with torch.no_grad():
            # state to ids
            encoded_ids = state  # as a list
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # use_seq_cache:
            output_ids = self.seq_cache.get(encoded_ids)
            if output_ids is not None:
                return output_ids

            # generate code answer
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())
            with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                        f"Response with the code ONLY. No other explanation or words attached!\n") + input_prompt
            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=self.width, max_length=2048)
            sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})
            if len(log_probs) == 0:  # 之前已经生成了完整的程序,所以gpt判断不再需要token在后面
                log_probs = [{'<|endoftext|>': 1.0}]
            model_output = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0).to(self.device),
                                          scores=log_probs,
                                          attentions=None,
                                          hidden_states=None,
                                          beam_indices=None)

            # top_k_cache
            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)
            output_ids_list = model_output.sequences.tolist()
            output_ids = output_ids_list[0]

            # seq_cache and time_stamps
            self.seq_cache.add(encoded_ids, output_ids)
            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_rationale_predict(self, state, with_verbal=False):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            if not with_verbal:

                with_instru_input_prompt = (f"{input_prompt} \n\n"
                                            f"Above is a problem to be solved by Python program. \n"
                                            f"I need you analyze this problem and provide strategies.  "
                                            f"I need you to output {self.width} possible thoughts and strategies. Remember each only contain one possible strategy of the problem.\n"
                                            f"Please wrap your response into a JSON object that contains keys `Thought-i` with i as the number of your thought, and key `Reasonableness` with the Reasonableness of each thought, which should between 0~1 and the sum should be 1. \n"
                                            f"The JSON should be a **list of dicts**, the dicts are splited with comma ','. \n"
                                            f"Example Answers:\n")
                with_instru_input_prompt += """
[
    {"Thought-1":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7},
    {"Thought-2":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29},
    {"Thought-3":" The problem can't be solved by Python.", "Reasonableness": 0.01}
]
    """

                print('\n-----------------1')
                print(with_instru_input_prompt)

                if self.args.json_save_all:
                    if self.args.rollout_count == -1:
                        self.save_mid_json.append(f"expansion_input_root: \n{with_instru_input_prompt}")
                    else:
                        self.save_mid_json.append(f"expansion_input_{self.args.rollout_count}: \n{with_instru_input_prompt}")


                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                print('\n-----------------2')
                print(response_text)

                if self.args.json_save_all:
                    if self.args.rollout_count == -1:
                        self.save_mid_json.append(f"expansion_output_root: \n{response_text}")
                    else:
                        self.save_mid_json.append(f"expansion_output_{self.args.rollout_count}: \n{response_text}")

            else:

                with_instru_input_prompt = (f"{input_prompt} \n"
                                            f"I need you to analyze and provide new thoughts that can lead to the correct solution code. \n"
                                            f"The goal is that the thoughts could lead to the code that not only avoids the current error but also solve the problem in a way that handles other potential test cases that we haven't encountered yet. \n"
                                            f"I need you to output {self.width} possible thoughts and strategies. Remember each only contain one possible strategy of the problem.\n"
                                            f"Please wrap your response into a JSON object that contains keys `Thought-i` with i as the number of your thought, and key `Reasonableness` with the Reasonableness of each thought, which should between 0~1 and the sum should be 1.\n"
                                            f"The JSON should be a **list of dicts**, the dicts are splited with comma ','. \n"
                                            f"Example Answers:\n")
                with_instru_input_prompt += """
[
    {"Thought-1":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7},
    {"Thought-2":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29},
    {"Thought-3":" The problem can't be solved by Python.", "Reasonableness": 0.01}
]
                """

                print('\n-----------------1')
                print(with_instru_input_prompt)
                prompt_lengh = len(self.tokenizer.encode(with_instru_input_prompt, allowed_special={'<|endoftext|>'}))
                print(f"Expansion Input Prompt length: {prompt_lengh}")

                if self.args.json_save_all:
                    if self.args.rollout_count == -1:
                        self.save_mid_json.append(f"expansion_input_root: \n{with_instru_input_prompt}")
                    else:
                        self.save_mid_json.append(f"expansion_input_{self.args.rollout_count}: \n{with_instru_input_prompt}")

                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                print('\n-----------------2')
                print(response_text)
                output_length = len(self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'}))
                print(f"Expansion Output length: {output_length}")

                if self.args.json_save_all:
                    if self.args.rollout_count == -1:
                        self.save_mid_json.append(f"expansion_output_root: \n{response_text}")
                    else:
                        self.save_mid_json.append(f"expansion_output_{self.args.rollout_count}: \n{response_text}")


            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            try:
                self.args.all_json_num += 1
                if response_text.strip()[0] != '[':
                    response_text = '[' + response_text + ']'

                response_text = json.loads(response_text)
                top_scores = []
                top_lines = []
                top_lines_text = []
                for i, ele in enumerate(response_text):
                    top_scores.append(ele['Reasonableness'])
                    ele[f'Thought-{i + 1}'] = '\nThought: ' + ele[f'Thought-{i + 1}']
                    top_lines.append(self.tokenizer.encode(ele[f'Thought-{i + 1}'], allowed_special={'<|endoftext|>'}))
                    top_lines_text.append(ele[f'Thought-{i + 1}'])
            except Exception as e:
                self.args.failed_json_num += 1
                top_lines = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]

            return top_lines, top_scores

    def get_rationale_predicted_sequence(self, state, horizon=None, renewchild_count=0):
        with torch.no_grad():
            encoded_ids = state  # as a list
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # use_seq_cache:
            output_ids = self.seq_cache.get(encoded_ids)
            if output_ids is not None:
                return output_ids

            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                        f"Also some thoughts are included that you can refer to and build upon when writing the code. "
                                        f"Answer with the code ONLY. No other explanation or words attached!\n") + input_prompt

            print('\n-----------------3')
            print(with_instru_input_prompt)
            if self.args.json_save_all:
                self.save_mid_json.append(f"simulation_input_{self.args.rollout_count}_{renewchild_count}: \n{with_instru_input_prompt}")

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024)

            print('\n-----------------4')
            print(response_text)
            if self.args.json_save_all:
                self.save_mid_json.append(f"simulation_output_{self.args.rollout_count}_{renewchild_count}: \n{response_text}")

            sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})
            model_output = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0).to(self.device),
                                          scores=log_probs,
                                          attentions=None,
                                          hidden_states=None,
                                          beam_indices=None)

            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(encoded_ids, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def get_top_k_codes_predict(self, state, with_verbal=False):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            if not with_verbal:
                with_instru_input_prompt = (f"{input_prompt} \n\n"
                                            f"Above is your previous implementation of a function, a series of unit tests results, and your self-reflection on your previous implementation. \n"
                                            f"I need you analyze this problem and provide solution codes."
                                            f"I need you to output {self.width} possible code solutions. Remember each only contain one possible solution of the problem.\n"
                                            f"Please wrap each solution into ```python ... ``` format and each titles with `Solution-i` with i as the index.\n"
                                            f"Each solution should contain the complete program including all the imports and function header in your response, and each solution should has a distinct strategy.\n"
                                            f"Example Answers:\n")
                with_instru_input_prompt += """
Solution 1:
```python
def add(a, b):
    return a + b
```

Solution 2:
```python
def add(a, b):
    c = a + b
    return c
```

Solution 3:
```python
def add(a, b):
    c = a + b
    print(c)
```
"""

                print('\n-----------------1')
                print(with_instru_input_prompt)

                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                print('\n-----------------2')
                print(response_text)
            else:
                with_instru_input_prompt = (f"{input_prompt} \n\n"
                                            f"I need you analyze this problem and provide solution codes."
                                            f"I need you to output {self.width} possible code solutions. Remember each only contain one possible solution of the problem.\n"
                                            f"Please wrap each solution into ```python ... ``` format and each titles with `Solution-i` with i as the index.\n"
                                            f"Each solution should contain the complete program including all the imports and function header in your response, and each solution should has a distinct strategy.\n"
                                            f"Example Answers:\n")
                with_instru_input_prompt += """
Solution 1:
```python
def add(a, b):
    return a + b
```

Solution 2:
```python
def add(a, b):
    c = a + b
    return c
```

Solution 3:
```python
def add(a, b):
    c = a + b
    print(c)
```
                """

                print('\n-----------------1')
                print(with_instru_input_prompt)

                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                print('\n-----------------2')
                print(response_text)

            try:
                code_blocks = extract_python_code(response_text)
                top_scores = []
                top_codes = []
                top_code_text = []
                for i, ele in enumerate(code_blocks):
                    top_scores.append(1.0)
                    top_codes.append(self.tokenizer.encode(ele, allowed_special={'<|endoftext|>'}))
                    top_code_text.append(ele)
            except Exception as e:
                top_codes = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]

            return top_codes, top_scores

    def get_code_predicted_sequence(self, state, horizon=None):
        trajectory = self.tokenizer.decode(state)
        code_text = extract_python_code(trajectory)[-1]

        print('\n-----------------3 extract code input')
        print(trajectory)

        print('\n-----------------4 extract code output')
        print(code_text)

        sequences = self.tokenizer.encode(code_text, allowed_special={'<|endoftext|>'})
        self.time_stamps.append(time.time())
        return sequences

    def get_top_k_new_action_predict(self, state, with_verbal=False):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # 生成下面的line，以及line level的概率
            input_prompt = self.tokenizer.decode(input_ids[0].tolist())
            if not with_verbal:
                with_instru_input_prompt = (f"{input_prompt} \n\n"
                                            f"Above is a problem to be solved by Python program. \n"
                                            f"I need you analyze this problem and provide solving strategies."
                                            f"I need you to output {self.width} possible thoughts and strategies. Remember each only contain one possible strategy of the problem.\n"
                                            f"Please wrap your response into a JSON object that contains keys `Thought-i` with i as the number of your thought, and key `Reasonableness` with the Reasonableness of each thought. \n"
                                            f"The JSON should be a **list of dicts**, the dicts are split with comma ','. \n"
                                            f"Example Answers:\n")
                with_instru_input_prompt += """
[
    {"Thought-1":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7},
    {"Thought-2":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29},
    {"Thought-3":" The problem can't be solved by Python.", "Reasonableness": 0.01}
]
"""

                print('\n-----------------1')
                print(with_instru_input_prompt)

                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                print('\n-----------------2')
                print(response_text)
            else:
                with_instru_input_prompt = (f"{input_prompt} \n"  # The thoughts should be mainly about the error?
                                            f"The code generated failed on some test cases. The errors or in-correct outputs are given above. \n"
                                            f"I need you to analyze and provide thoughts for the programmer that can lead to the correct solution code. \n"
                                            f"The goal is that the thoughts could lead to the code that not only avoids the current error but also solve the problem in a way that handles other potential test cases that we haven't encountered yet. \n"
                                            f"I need you to output {self.width} possible thoughts. Remember each only contain one possible modifying suggestion to correct the error and solve the problem.\n"
                                            f"Please wrap your response into a JSON object that contains keys `Thought-i` with i as the number of your thought, and key `Reasonableness` with the Reasonableness of each thought.\n"
                                            f"The JSON should be a **list of dicts**, the dicts are split with comma ','. \n"
                                            f"Example Answers:\n")
                with_instru_input_prompt += """
[
    {"Thought-1":" We could use the print function to finish the task in one line: print(2 + 3)", "Reasonableness": 0.7},
    {"Thought-2":" We should calculate the problem by setting a=2+3, and then print(a)", "Reasonableness": 0.29},
    {"Thought-3":" The problem can't be solved by Python.", "Reasonableness": 0.01}
]
"""

                print('\n-----------------1')
                print(with_instru_input_prompt)
                prompt_lengh = len(self.tokenizer.encode(with_instru_input_prompt, allowed_special={'<|endoftext|>'}))
                print(f"Expansion Input Prompt length: {prompt_lengh}")

                response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024, temperature=0.0)
                print('\n-----------------2')
                print(response_text)
                output_length = len(self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'}))
                print(f"Expansion Output length: {output_length}")

            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]

            try:
                if response_text.strip()[0] != '[':
                    response_text = '[' + response_text + ']'

                response_text = json.loads(response_text)
                top_scores = []
                top_lines = []
                top_lines_text = []
                for i, ele in enumerate(response_text):
                    top_scores.append(ele['Reasonableness'])
                    ele[f'Thought-{i + 1}'] = '\nThought: ' + ele[f'Thought-{i + 1}']
                    top_lines.append(self.tokenizer.encode(ele[f'Thought-{i + 1}'], allowed_special={'<|endoftext|>'}))
                    top_lines_text.append(ele[f'Thought-{i + 1}'])
            except Exception as e:
                top_lines = [self.tokenizer.encode('\n', allowed_special={'<|endoftext|>'}) for i in range(self.width)]
                top_scores = [1.0 for i in range(self.width)]

            return top_lines, top_scores
    def get_new_action_predicted_sequence(self, state, horizon=None):
        with torch.no_grad():
            encoded_ids = state  # as a list
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # use_seq_cache:
            output_ids = self.seq_cache.get(encoded_ids)
            if output_ids is not None:
                return output_ids

            input_prompt = self.tokenizer.decode(input_ids[0].tolist())

            with_instru_input_prompt = (f"Complete the Python program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n"
                                        f"Some thoughts are included that you can refer to and build upon when writing the code. "
                                        f"Answer with the code ONLY. No other explanation or words attached!\n") + input_prompt

            print('\n-----------------3')
            print(with_instru_input_prompt)

            response_text, log_probs = self.generate_response_api(with_instru_input_prompt, top_k=1, max_length=1024)

            print('\n-----------------4')
            print(response_text)

            sequences = self.tokenizer.encode(response_text, allowed_special={'<|endoftext|>'})
            model_output = WithProbReturn(sequences=torch.tensor(sequences).unsqueeze(0).to(self.device),
                                          scores=log_probs,
                                          attentions=None,
                                          hidden_states=None,
                                          beam_indices=None)

            if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)

            output_ids_list = model_output.sequences.tolist()

            output_ids = output_ids_list[0]

            # use_seq_cache
            self.seq_cache.add(encoded_ids, output_ids)

            self.time_stamps.append(time.time())
            return output_ids

    def clean_cache(self):
        self.top_k_cache = GPTTopKCache(self.args.width, cache_steps=self.args.top_k_cache_steps, tokenizer=self.tokenizer, args=self.args)
        self.seq_cache = GPTSeqCache(args=self.args)
        self.time_stamps = []


class WithProbReturn:
    def __init__(self, sequences, scores, attentions, hidden_states, beam_indices=None, top_tokens=None):
        self.sequences = sequences
        self.scores = scores
        self.attentions = attentions
        self.hidden_states = hidden_states
        self.beam_indices = beam_indices
        self.top_tokens = top_tokens


def extract_python_code(text):
    pattern = r'```python(.*?)```'
    code_blocks = re.findall(pattern, text, re.DOTALL)
    for i in range(len(code_blocks)):
        code_blocks[i] = code_blocks[i].strip()
        code_blocks[i] = '\n```python \n' + code_blocks[i] + '\n```\n'
    return code_blocks