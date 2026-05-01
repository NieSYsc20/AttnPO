import torch
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score.adapt_think_rm import adapt_think_rm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AccRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine = 0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = adapt_think_rm  

    def __call__(self, data: DataProto, return_dict=False):

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        all_scores = []  
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum().item()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
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
            all_scores.append(score)   
            
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            score = all_scores[i]
            reward = score["score"]
            for key, value in score.items():
                reward_extra_info[key].append(value)
            reward_tensor[i, score["response_length"] - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    

