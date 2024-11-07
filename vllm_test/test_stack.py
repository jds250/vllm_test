from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import modify_list,prefill_attn_event_list,prefill_mlp_event_list,decode_attn_event_list,decode_mlp_event_list,decode_time_event_list,prefill_energy
import torch
import vllm
from pynvml import *
from utils import data_utils
import os
import time
# su -c "/home/jds/nsight-systems-2024.6.1/bin/nsys profile -o test_file  --gpu-metrics-devices=all --enable nvml_metrics,-i10000 /home/jds/anaconda3/envs/work/bin/python vllm_test/attn_test.py"

# model_name_0 = "/home/jds/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"

def calculate_total_time(event_list):
    total_time = 0.0
    for start_event, stop_event in event_list:
        time = start_event.elapsed_time(stop_event)  # 获取两个事件之间的时间（单位：毫秒）
        total_time += time
    return total_time


os.environ['VLLM_ATTENTION_BACKEND']= "XFORMERS"
os.environ['CUDA_VISIBLE_DEVICES']="1"

prompts = data_utils.generate_prompts_from_dataset("wikitext", "wikitext-103-v1", num_prompts=12, target_length=1024,tokenizer_name="/home/jds/test/vllm_test/llama2-7b")

# gpu_device = torch.device("cuda:0")
# torch.cuda.set_device(gpu_device)

sampling_params = SamplingParams(temperature=0, top_p=0.95,ignore_eos=True,max_tokens=21)

print("-----INIT-----")
llm = LLM(model="/home/jds/test/vllm_test/llama2-7b",enable_prefix_caching=False,enforce_eager=True,tensor_parallel_size=1)


outputs = llm.generate(prompts, sampling_params)

