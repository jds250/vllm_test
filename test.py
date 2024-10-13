import torch
import time
from pynvml import *
import matplotlib.pyplot as plt
# /home/liangjy53/test/testing/llama2-7b
import json
from vllm_test.utils.data_utils import load_data,plot_data

if __name__ == '__main__':

    data = load_data("/home/jds/test/vllm_test/data.pkl")
    
    plot_data(data=data)