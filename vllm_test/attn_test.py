from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import modify_list,prefill_attn_event_list,prefill_mlp_event_list,decode_attn_event_list,decode_mlp_event_list,decode_time_event_list,decode_energy
import torch
import vllm
from pynvml import *
from utils import data_utils
import os
import time
import multiprocessing as mp
from typing import List, Tuple
# su -c "/home/jds/nsight-systems-2024.6.1/bin/nsys profile -o test_file  --gpu-metrics-devices=all --enable nvml_metrics,-i10000 /home/jds/anaconda3/envs/work/bin/python vllm_test/attn_test.py"

# model_name_0 = "/home/jds/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def calculate_total_time(event_list):
    total_time = 0.0
    for start_event, stop_event in event_list:
        time = start_event.elapsed_time(stop_event)  # 获取两个事件之间的时间（单位：毫秒）
        total_time += time
    return total_time

def calculate_energy(event_list,power_list):
    total_time = []
    for start_event, stop_event in event_list:
        time = start_event.elapsed_time(stop_event)  # 获取两个事件之间的时间（单位：毫秒）
        total_time.append(time)
    result = [a * b for a, b in zip(total_time, power_list)]
    return sum(result)


os.environ['VLLM_ATTENTION_BACKEND']= "XFORMERS"
# os.environ['CUDA_VISIBLE_DEVICES']="1"

prompts = data_utils.generate_prompts_from_dataset("wikitext", "wikitext-103-v1", num_prompts=12, target_length=1024,tokenizer_name="/home/jds/test/vllm_test/llama2-7b")

# gpu_device = torch.device("cuda:0")
# torch.cuda.set_device(gpu_device)

sampling_params = SamplingParams(temperature=0, top_p=0.95,ignore_eos=True,max_tokens=21)

print("-----INIT-----")
llm = LLM(model="/home/jds/test/vllm_test/llama2-7b",enable_prefix_caching=False,enforce_eager=True,tensor_parallel_size=2)

# nvml init
nvmlInit()
device_handle = nvmlDeviceGetHandleByIndex(1)
device2_handle = nvmlDeviceGetHandleByIndex(0)

mem_clock = nvmlDeviceGetSupportedMemoryClocks(device_handle)
nvmlDeviceSetMemoryLockedClocks(device_handle,mem_clock[0],mem_clock[0])
supported_gpu_clocks = nvmlDeviceGetSupportedGraphicsClocks(device_handle, mem_clock[0])
gpu_clocks = list( filter(lambda x: x <= 1900 and x >= 700 and (x-210)%60==0, supported_gpu_clocks))

prefill_time = []
decode_time = []
prefill_mlp_time = []
decode_mlp_time = []
decode_time_full = []
power = []

gpu_energy_efficiency=[]
gpu_memband_util = []

# warm up
nvmlDeviceResetGpuLockedClocks(device_handle)
nvmlDeviceResetGpuLockedClocks(device2_handle)

print("------warm up------")
for _ in range(4):
    outputs = llm.generate(prompts, sampling_params)

# a = nvmlDeviceGetUtilizationRates(device_handle)

# for gpu_clock in gpu_clocks:
#     nvmlDeviceSetGpuLockedClocks(device_handle, gpu_clock, gpu_clock)

#     # current_power = nvmlDeviceGetPowerUsage(device_handle)
#     current_power = nvmlDeviceGetClockInfo(device_handle,1)
#     power.append(current_power)

modify_list()
k=[500,1000,2000]
for i in k:
    
    if i == 2:
        continue
    print("-----Prompts Generation-----")
    prompts = data_utils.generate_prompts_from_dataset("wikitext", "wikitext-103-v1", num_prompts=16, target_length=1*(i+1),tokenizer_name="/home/jds/test/vllm_test/llama2-7b")
    temp_ef = []
    temp_de = []
    temp_e = []
    temp_p = []
    for gpu_clock in gpu_clocks:
        modify_list()
        
        nvmlDeviceSetGpuLockedClocks(device_handle, gpu_clock, gpu_clock)
        nvmlDeviceSetGpuLockedClocks(device2_handle, gpu_clock, gpu_clock)

        energy_start = nvmlDeviceGetTotalEnergyConsumption(device_handle)
        
        m = 1

        for _ in range(m):
            outputs = llm.generate(prompts, sampling_params)
        # prefill_time.append(calculate_total_time(prefill_attn_event_list)/k)
            # print(nvmlDeviceGetClockInfo(device_handle,1))
        # decode_time.append(calculate_total_time(decode_attn_event_list))
        # decode_time_full.append(calculate_total_time(decode_time_event_list))
        # prefill_mlp_time.append(calculate_total_time(prefill_mlp_event_list)/k)
        # decode_mlp_time.append(calculate_total_time(decode_mlp_event_list)/k)
        torch.cuda.synchronize()

        energy_end = nvmlDeviceGetTotalEnergyConsumption(device_handle)
        temp_de.append(calculate_total_time(decode_time_event_list)/m)
        temp_e.append(calculate_energy(decode_time_event_list,decode_energy))
        perf = 1/calculate_total_time(decode_time_event_list)
        avg_power = sum(decode_energy)/len(decode_energy)/1000
        temp_ef.append(perf / avg_power)
        temp_p.append(sum(decode_energy)/len(decode_energy))

    power.append(temp_p)
    temp_freqmax = temp_ef[0]
    temp_ef = [x/temp_freqmax for x in temp_ef]
    decode_time.append(temp_de)
    gpu_energy_efficiency.append(temp_ef)
    # print("time:",end-start," ","energy:",energy_end-energy_start)

# data_utils.save_data(prefill_mlp_time=prefill_mlp_time,prefill_time=prefill_time,decode_mlp_time=decode_mlp_time,decode_time=decode_time,power=power)

# x = [256*(i+1) for i in range(k)]

print("gpu_clocks=",gpu_clocks)
# print("X=",x)

print("energy_efficiency=",gpu_energy_efficiency)
print("decode_time=",decode_time)
print("decode_energy=",power)

nvmlDeviceResetGpuLockedClocks(device_handle)
nvmlDeviceResetMemoryLockedClocks(device_handle)

nvmlDeviceResetGpuLockedClocks(device2_handle)
nvmlDeviceResetMemoryLockedClocks(device2_handle)
# nvml shutdown
nvmlShutdown()