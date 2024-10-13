import matplotlib.pyplot as plt
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer

def generate_prompts_from_dataset(dataset_name, subset_name=None, split="train", num_prompts=4, target_length=512, tokenizer_name="gpt2"):
    """
    从指定的数据集中生成经过 tokenizer 处理后长度为 target_length 的 prompts。
    如果文本太长则截断，太短则拼接下一个文本直到满足长度要求。

    参数:
    - dataset_name: 数据集的名称，例如 "wikitext"。
    - subset_name: 如果数据集有子集，指定子集的名称，例如 "wikitext-103-v1"。
    - split: 数据集的划分，默认是 "train"。
    - num_prompts: 生成的 prompt 数量，默认是 4。
    - target_length: 每个 prompt 的目标 token 长度，默认是 512。
    - tokenizer_name: tokenizer 的名称，默认使用 "gpt2"。

    返回:
    - prompts: 一个包含生成的 prompts 的列表，每个 prompt 的 token 数量为 target_length。
    """

    # 加载数据集
    if subset_name:
        dataset = load_dataset(dataset_name, subset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 初始化 prompts 列表
    prompts = []

    # 初始化一个文本缓冲区，用于拼接短文本
    buffer_text = ""

    # 遍历数据集，生成符合长度要求的 prompts
    for i, item in enumerate(dataset):
        buffer_text += item['text'].strip() + " "  # 先把当前文本添加到缓冲区
        
        # 对缓冲区的文本进行 tokenization
        tokens = tokenizer(buffer_text, truncation=False, add_special_tokens=False)["input_ids"]
        
        # 如果 token 数量达到或超过 target_length，则生成 prompt
        if len(tokens) >= target_length:
            truncated_tokens = tokens[:target_length]  # 截断到 target_length
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            prompts.append(truncated_text)
            
            # 清空缓冲区，将多余的 token 放入缓冲区
            remaining_tokens = tokens[target_length:]
            buffer_text = tokenizer.decode(remaining_tokens, skip_special_tokens=True)
        
        # 如果生成的 prompts 数量已满足要求，跳出循环
        if len(prompts) >= num_prompts:
            break

    return prompts

# 保存数据到文件
def save_data(prefill_time, decode_time, prefill_mlp_time, decode_mlp_time, power, filename='/home/jds/test/vllm_test/data.pkl'):
    """
    保存五组数据到本地文件。
    """
    data = {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "prefill_mlp_time": prefill_mlp_time,
        "decode_mlp_time": decode_mlp_time,
        "power": power
    }
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"数据已保存到 {filename}")

# 从文件读取数据
def load_data(filename='data.pkl'):
    """
    从本地文件读取数据。
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"从 {filename} 中读取数据成功")
    return data

# 画出时间与功耗的关系图
def plot_data(data):
    """
    绘制 prefill_time, decode_time, prefill_mlp_time, decode_mlp_time 与 power 的关系图。
    将 decode attention 和 decode mlp 绘制在同一个图中。
    """
    power = data['power']
    
    plt.figure(figsize=(10, 8))

    # Prefill Attention Time vs Power
    plt.subplot(2, 2, 1)
    plt.plot(power, data['prefill_time'], marker='o', label='Prefill Attention Time')
    plt.title('Prefill Attention Time vs Power')
    plt.xlabel('Power')
    plt.ylabel('Prefill Time')

    # Prefill MLP Time vs Power
    plt.subplot(2, 2, 2)
    plt.plot(power, data['prefill_mlp_time'], marker='o', label='Prefill MLP Time')
    plt.title('Prefill MLP Time vs Power')
    plt.xlabel('Power')
    plt.ylabel('Prefill MLP Time')

    # Decode Attention Time 和 Decode MLP Time vs Power
    plt.subplot(2, 2, 3)
    plt.plot(power, data['decode_time'], marker='o', label='Decode Attention Time')
    plt.plot(power, data['decode_mlp_time'], marker='x', label='Decode MLP Time')
    plt.title('Decode Attention & MLP Time vs Power')
    plt.xlabel('Power')
    plt.ylabel('Decode Time')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("/home/jds/test/power.jpg")