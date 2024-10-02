# -*- coding:utf-8 _*-
import torch


class GPTTopKCache:
    def __init__(self, k, cache_steps, tokenizer, args):
        """
        A cache dict, self.cache[input_ids] = [the i-th most likely token output, for i in range(k)]
        """
        self.k = k
        self.cache_steps = cache_steps
        self.tokenizer = tokenizer

        self.cache = {}
        self.args = args

    def add(self, input_ids, output_ids, scores, beam_indices=None):
        """
        Args:
            input_ids: input sequence, which is the problem description
            output_ids: the complete generated program
            scores: scores at each generation step

        Returns:
            None
        """
        output_ids = output_ids.tolist()
        if beam_indices is not None:
            beam_indices = beam_indices.tolist()

        prefix_len = len(input_ids[0])
        # maybe do not need to cache all the way?
        output_len = len(output_ids[0])
        if output_len == prefix_len + len(scores):
            token_range = range(prefix_len, output_len)
        else:
            token_range = range(0, output_len)
        # assert len(scores) == len(token_range)
        if len(scores) != len(token_range):
            print('len(scores) != len(token_range)', len(scores), len(token_range))
            return

        if len(token_range) > prefix_len + self.cache_steps:
            if output_len == prefix_len + len(scores):
                token_range = range(prefix_len, prefix_len + self.cache_steps)
            else:
                token_range = range(0, prefix_len + self.cache_steps)

        for idx, end_index in enumerate(token_range):  # for each token in the output
            for batch_idx in range(len(output_ids)):
                if beam_indices is not None:
                    beam_idx = beam_indices[batch_idx][idx]
                    if beam_idx != batch_idx: continue  # fix me

                key = tuple(output_ids[batch_idx][:end_index])

                if key in self.cache.keys():
                    # already stored, possible because this prefix is expanded more than once in beam search
                    continue

                if self.args.arch == 'gpt3.5completion':  # gpt3.5 completion
                    top_scores = []
                    top_tokens = []
                    for top_token in scores[0].keys():
                        top_scores.append(scores[0][top_token])
                        top_tokens.append(self.tokenizer.encode(top_token, allowed_special={'<|endoftext|>'})[0])
                    return top_tokens, top_scores
                elif self.args.arch in ['gpt3.5', 'gpt4', 'gpt4o-mini', 'gpt4o']:  # gpt3.5
                    top_scores = []
                    top_tokens = []
                    for token_probs in scores[idx].top_logprobs:
                        top_scores.append(token_probs.logprob)
                        top_tokens.append(self.tokenizer.encode(token_probs.token, allowed_special={'<|endoftext|>'})[0])
                    self.cache[key] = (top_tokens, top_scores)

                    # print('batch idx', batch_idx)
                # print('input', self.tokenizer.decode(key[132:]))
                # print('top k', [self.tokenizer.decode(token) for token in top_k_tokens])
                # print()

                # if key in self.cache.keys():
                #     assert top_k_tokens.tolist() == self.cache[key][0],\
                #         (self.tokenizer.decode(key[132:]), self.tokenizer.decode(top_k_tokens), self.tokenizer.decode(self.cache[key][0]))

    def get(self, input_ids):
        input_ids = tuple(input_ids)

        if input_ids in self.cache.keys():
            return self.cache[input_ids]
        else:
            return None

    def clear(self, encoded_ids=None):
        if encoded_ids is None:
            # clear cache unconditionally
            self.cache = {}
        else:
            encoded_ids = tuple(encoded_ids)
            keys_to_remove = []
            for cached_key in self.cache.keys():
                if cached_key[:len(encoded_ids)] != encoded_ids:
                    keys_to_remove.append(cached_key)
            for k in keys_to_remove: del self.cache[k]


class GPTSeqCache:
    def __init__(self, args):
        self.cache = {}
        self.args = args

    def add(self, query_ids, output_ids):
        query_ids = tuple(query_ids)
        self.cache[query_ids] = output_ids

    def get(self, query_ids):
        for input_ids, output_ids in self.cache.items():
            if query_ids == output_ids[:len(query_ids)]:
                return output_ids

        return None

    def clear(self, new_state):
        self.cache = {input_ids: output_ids for input_ids, output_ids in self.cache.items() if new_state == output_ids[:len(new_state)]}
