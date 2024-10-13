from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import modify_list,prefill_attn_event_list,prefill_mlp_event_list,decode_attn_event_list,decode_mlp_event_list
import torch
import vllm
from pynvml import *
from utils import data_utils
import os

def calculate_total_time(event_list):
    total_time = 0.0
    for start_event, stop_event in event_list:
        time = start_event.elapsed_time(stop_event)  # 获取两个事件之间的时间（单位：毫秒）
        total_time += time
    return total_time


os.environ['VLLM_ATTENTION_BACKEND']= "XFORMERS"
prompts = data_utils.generate_prompts_from_dataset("wikitext", "wikitext-103-v1", num_prompts=16, target_length=1024,tokenizer_name="/home/jds/test/vllm_test/llama2-7b")

gpu_device = torch.device("cuda:0")
torch.cuda.set_device(gpu_device)

sampling_params = SamplingParams(temperature=0, top_p=0.95,ignore_eos=True,max_tokens=16)

llm = LLM(model="/home/jds/test/vllm_test/llama2-7b",enable_prefix_caching=False,enforce_eager=True)

modify_list()

# nvml init
nvmlInit()
device_handle = nvmlDeviceGetHandleByIndex(0)
mem_clock = nvmlDeviceGetSupportedMemoryClocks(device_handle)
nvmlDeviceSetMemoryLockedClocks(device_handle,mem_clock[0],mem_clock[0])
supported_gpu_clocks = nvmlDeviceGetSupportedGraphicsClocks(device_handle, mem_clock[0])
gpu_clocks = list( filter(lambda x: x <= 1900 and x >= 700 and (x-210)%60==0, supported_gpu_clocks))

prefill_time = []
decode_time = []
prefill_mlp_time = []
decode_mlp_time = []
power = []
# warm up
outputs = llm.generate(prompts, sampling_params)

for gpu_clock in gpu_clocks:
    nvmlDeviceSetGpuLockedClocks(device_handle, gpu_clock, gpu_clock)
    current_power = nvmlDeviceGetClockInfo(device_handle,1)
    power.append(current_power)

    modify_list()
    k=1
    for _ in range(k):
        outputs = llm.generate(prompts, sampling_params)

        prefill_time.append(calculate_total_time(prefill_attn_event_list)/k)
        decode_time.append(calculate_total_time(decode_attn_event_list)/k)
        prefill_mlp_time.append(calculate_total_time(prefill_mlp_event_list)/k)
        decode_mlp_time.append(calculate_total_time(decode_mlp_event_list)/k)

data_utils.save_data(prefill_mlp_time=prefill_mlp_time,prefill_time=prefill_time,decode_mlp_time=decode_mlp_time,decode_time=decode_time,power=power)

nvmlDeviceResetGpuLockedClocks(device_handle)
nvmlDeviceResetMemoryLockedClocks(device_handle)
# nvml shutdown
nvmlShutdown()