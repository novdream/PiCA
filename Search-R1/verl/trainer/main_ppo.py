# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import requests
import torch.nn.functional as F
import random
def process_qwen_response(sequences):
    response = sequences.split("<|im_start|>assistant")[-1]
    # question = seq.split("<|im_end|>")[0]
    return response
def process_llama_question(sequences):
    seq = sequences.split("Question:")[-1]
    question = seq.split("<|eot_id|>")[0]
    return question
def process_qwen_question(sequences):
    seq = sequences.split("Question:")[-1]
    question = seq.split("<|im_end|>")[0]
    return question
def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")
    
def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'musi','bamboogle']:
        return qa_em.compute_score_f1
        # return qa_em.compute_score_em
    else:
        raise NotImplementedError
def _extract_qwen_solution(sequences_str):
    # Extract the solution from the sequences string
    
    # The solution is the string after the first occurrence of "Assistant"
    return sequences_str.split("<|im_start|>assistant")[-1].strip() 
def _extract_llama_solution(sequences_str):
    # Extract the solution from the sequences string
    
    # The solution is the string after the first occurrence of "Assistant"
    return sequences_str.split("assistant<|end_header_id|>")[-1].strip() 
def compact_text(text: str) -> str:
    """
    去除 \n 并压缩空格，使文本紧凑
    """
    text = text.replace("\n", " ").replace("\r", " ")

    text = re.sub(r'\s+', ' ', text)
    

    return text.strip()
def process_sequences_robust(text, special_token):
    # 这里的 .*? 是核心，确保匹配到最近的结束标签就停止
    # 正则解释：
    # 1. (<warning>.*?</warning>) : 匹配并捕获警告标签（禁区）
    # 2. | : 或者
    # 3. (</search>)(?=\s*<information>) : 匹配 </search>，
    #    但后面必须跟着空白符和 <information>。(?=...) 是非捕获预查，不会消耗文本。
    pattern = r'(<warning>.*?</warning>)|(</search>)(?=\s*<information>)'
    
    def replacement_func(match):
        # 如果命中了第一个分组（warning 标签），原样返回
        if match.group(1):
            return match.group(1)
        
        # 否则，说明命中了符合条件的 </search>
        # match.group(2) 就是 "</search>"
        return f"{match.group(2)}{special_token}\n"

    # 使用 re.sub 直接完成替换
    return re.sub(pattern, replacement_func, text, flags=re.DOTALL)

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., vllm_api_url=None,  special_token="<|vision_start|>", step_reward_scale=0.6, baseline_step_reward=0.5,outcome_reward_scale=1.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.vllm_api_url = vllm_api_url  
        self.step_reward_scale = step_reward_scale
        self.special_token = special_token
        # self.special_token_id = convert_token_to_id(self.special_token, self.tokenizer)
        self.baseline_step_reward = baseline_step_reward
        self.outcome_reward_scale = outcome_reward_scale
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        answer_correct = [0] * len(data)
        ems = [0] * len(data)
        # search_key_score = [0] * len(data)
        step_scores = [0] * len(data)
        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            # print(sequences_str)
            # quit(0)
            if "<|im_start|>assistant" in sequences_str:
                sequences_str = _extract_qwen_solution(sequences_str)
            if "<|start_header_id|>assistant" in sequences_str:
                sequences_str = _extract_llama_solution(sequences_str)
                
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            if 'step_scores' in score:
                score['step_scores'] = self.comupute_step_reward(valid_prompt_ids,valid_response_ids)
                reward_tensor[i] += self.step_reward_scale * self._compute_step_reward_tensor(reward_tensor[i], valid_response_ids, score['step_scores'])
                step_scores[i] = sum(score['step_scores']) if len(score['step_scores']) > 0 else 0.0

            
            print('outcome reward:',score['score'])
            reward_tensor[i, valid_response_length - 1] = self.outcome_reward_scale * (score['score'])
            answer_correct[i] = score['answer_correct']
            ems[i] = score['em']
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return reward_tensor, answer_correct, ems , step_scores
    
    def _get_rm_values(self, sentences: list[str]) -> list[float]:
        if type(sentences) == str:
            sentences = [sentences]
        payload = {
            "query": sentences,
            "prompts": [] 
        }
        response = requests.post(self.vllm_api_url, json=payload, timeout=30,proxies={"http": None, "https": None})
        
        # 检查 HTTP 状态码
        if response.status_code != 200:
            print("请求失败")
            print(f"错误信息: {response.text}")
            return None
            
        rewards = response.json()
        return rewards['rewards']
    
    def comupute_step_reward(self, valid_prompt_ids, valid_response_ids):
        sequences = self.tokenizer.decode(torch.cat((valid_prompt_ids, valid_response_ids)))
        response = self.tokenizer.decode(valid_response_ids)
        response = process_qwen_response(response)
        # question = process_qwen_question(sequences) 
        question = process_llama_question(sequences)
        sequences = question + response
        sequences = compact_text(sequences)
        modified_str = process_sequences_robust(sequences, self.special_token)
        modified_str += self.special_token  
        # print(modified_str)
        step_rewards = self._get_rm_values(modified_str)[0]
        
        # rewards_tensor = torch.tensor(step_rewards)
        # step_rewards_tensor = F.logsigmoid(-rewards_tensor)
        # step_rewards = step_rewards_tensor.tolist()
        do_print = random.randint(1, 64) == 1
        steps = modified_str.split(self.special_token)
        if do_print:
            print(f"--------------------------------")
            for i, r in enumerate(step_rewards):
                print(f"Step output: {steps[i]}")
                print(f"Step reward: {r}")
           
        step_rewards = [1- reward -self.baseline_step_reward for reward in step_rewards]
        # step_rewards = [0.0] * len(step_rewards)

        return step_rewards
    
    def _compute_step_reward_tensor(self, reward_tensor, valid_response_ids, step_rewards):
      
        response = self.tokenizer.decode(valid_response_ids)
        
        search_end_indices = [m.start() + m.group().find('</search>') + len('</search>') for m in re.finditer(r'<search>.*?</search>\s*<information>.*?</information>', response, re.DOTALL)]
        search_end_positions = [len(self.tokenizer.encode(response[:search_end_index])) for search_end_index in search_end_indices]
        print('step rewards len:',len(step_rewards), 'search tag len:',len(search_end_positions))
        steps_num = min(len(step_rewards), len(search_end_positions))
        alpha = 0.3
        gamma = 1.5
        # alpha = 0.1
        # gamma = 1.0
        for i in range(steps_num):
            # penalty = i * 0.05
            penalty = 0 if i < 3 else alpha * (gamma ** (i - 2)) # step2
            # penalty = 0
            reward_tensor[search_end_positions[i]-1] = (step_rewards[i] - penalty)
        return reward_tensor
    


import ray
import hydra

import os
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    os.environ['RAY_TMPDIR'] = 'tmp'
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN','RAY_TMPDIR': 'tmp',}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, format_score=0., vllm_api_url=config.reward_model.url,  
                              special_token=config.reward_model.special_token, step_reward_scale=config.step_reward_scale, 
                              baseline_step_reward=config.baseline_step_reward, outcome_reward_scale=config.outcome_reward_scale)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1,format_score=0., vllm_api_url=config.reward_model.url,  
                                  special_token=config.reward_model.special_token, step_reward_scale=config.step_reward_scale, 
                                  baseline_step_reward=config.baseline_step_reward, outcome_reward_scale=config.outcome_reward_scale)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
