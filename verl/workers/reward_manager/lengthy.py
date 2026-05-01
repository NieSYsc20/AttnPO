from ast import ExtSlice
import torch
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score.adapt_think_rm import adapt_think_rm
from verl.workers.reward_manager.segement import segment_cot
# from verl.utils.reward_score import length_penalty
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LengthyRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, length_penalty_config, num_examine = 0) -> None:
        self.tokenizer = tokenizer
        self.length_penalty_config = length_penalty_config
        self.num_examine = num_examine  
        self.compute_score = adapt_think_rm  
        for key, value in length_penalty_config.items():
            print(f"==={key}: {value} ===")


    def _find_token_index_for_char(self, char_pos, token_mapping):
        for idx, (token, (start, end)) in enumerate(token_mapping):
            if start <= char_pos < end:
                return idx 

    def _get_response_token_mapping(self, response_str):
        encoding = self.tokenizer(response_str, return_offsets_mapping=True, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        token_positions = list(zip(tokens, encoding["offset_mapping"]))
        return token_positions

    def __call__(self, data: DataProto, epsilon = 1e-8, return_dict=False):

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        all_scores = []  
        all_responses = []
        all_prompt_lens = []
        all_prompts =  []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            score = self.compute_score(   # {score: , acc: , pred: }
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            is_nothinking = response_str.strip().startswith('</think>')
            score.update({   
                'response_length': valid_response_length,
                'is_nothinking': is_nothinking,
                "ground_truth": ground_truth,
            }) 
            
            all_responses.append(response_str)
            all_prompt_lens.append(valid_prompt_length)
            all_prompts.append(prompt_str)
            all_scores.append(score)


        id2acc = defaultdict(list)
        id2lens = defaultdict(list)
        for i in range(len(data)):
            score = all_scores[i]
            uid = data[i].non_tensor_batch['uid']
            id2acc[uid].append(score['acc'])
            if score['acc'] == 1:
                id2lens[uid].append(score['response_length'])

        for i in range(len(data)):
            score = all_scores[i]
            uid = data[i].non_tensor_batch['uid']
            mean_acc = float(np.mean(id2acc[uid]))
            correct_shorest_len = float(np.min(id2lens[uid]) if id2lens[uid] else -100)
            score.update({'mean_acc': mean_acc, 'correct_shorest_len': correct_shorest_len})


        if self.length_penalty_config.method == 'traineff': 
            self.traineff_coff = self.length_penalty_config.traineff_coff  
            id2mean = {}
            id2std = {}
            for uid in id2lens:
                if len(id2lens[uid]) != 0:
                    id2mean[uid] =sum(id2lens[uid]) / len(id2lens[uid])
                    id2std[uid] = float(np.std(id2lens[uid]))
            for i in range(len(data)):
                data_item = data[i]  
                score = all_scores[i]
                uid = data_item.non_tensor_batch['uid']
                if score['acc'] == 1:
                    relative_length = (score['response_length'] - id2mean[uid]) / (id2std[uid] + epsilon)
                    score['score'] = float(score['score'] * (1 - self.traineff_coff * sigmoid(relative_length)))


        
        if self.length_penalty_config.enable_fgreward:

            for i in range(len(data)):
                all_scores[i].update({"do_fgreward": True})
            
            for i in range(len(data)):
                score = all_scores[i]
                if score["acc"] == 0:
                    score.update({"do_fgreward": False})


            for i in range(len(data)):
                data_item = data[i]  
                uid = data_item.non_tensor_batch['uid']
                score = all_scores[i]
                response_str = all_responses[i]
                prompt_len = all_prompt_lens[i]
                if score["do_fgreward"]:
                    try:
                        segments = segment_cot(response_str)
                    except Exception as e:
                        print("[FGReward] segment_cot 执行异常")
                        print("Response:", response_str)
                        print("Error:", repr(e))
                        score["do_fgreward"] = False
                        subroll_token_positions = []
                        seg_type = []
                        step_num = -100
                        continue
                    if len(segments) == 0 or len(segments) == 1:
                        score["do_fgreward"] = False
                        subroll_token_positions = []
                        step_num = -100
                        seg_type =[]
                    else:
                        positions = []
                        seg_type = []
                        for seg_i ,seg in enumerate(segments): 
                            positions.append(seg["start"])
                            seg_type.append(seg["kind"])
                        positions.append(response_str.find("</think>")) 
                        seg_type.append(4)
                        response_token_mapping = self._get_response_token_mapping(response_str)
                        subroll_token_positions = [] 
                        for position in positions:
                            token_idx = self._find_token_index_for_char(position, response_token_mapping)
                            token_idx += prompt_len  
                            subroll_token_positions.append(token_idx)
                        subroll_token_positions = [0] + subroll_token_positions + [subroll_token_positions[-1] + 1] + [prompt_len + score['response_length']] 
                        seg_type = [4] + seg_type + [4] + [4]
                        step_num = len(subroll_token_positions) - 4
                else:
                    subroll_token_positions = []
                    seg_type = []
                    step_num = -100
                score.update({"subroll_token_positions": subroll_token_positions})
                score.update({"seg_type":seg_type})
                score.update({"step_num": step_num})
       
        for i in range(len(data)):
            score = all_scores[i]
            for key, value in score.items():
                reward_extra_info[key].append(value)
            reward_tensor[i, score["response_length"] - 1] = score['score']
        

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    



