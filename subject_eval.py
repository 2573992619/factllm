from llm_utils import add_prompt_chat,add_prompt_text,llm_init,llm_response,extract_answer
from datasets import load_dataset
from tqdm import tqdm
import os

import metrics

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# llama3_8B,'mistral-8B','DeepSeek-R1-7B'
model_name = 'DeepSeek-R1-7B'
dataset_path = '/media/sysu/computer2/mjn/factllm/data/MMLU/all/validation-00000-of-00001.parquet'
# dataset_path = '/media/sysu/computer2/mjn/factllm/data/MMLU/all/test-00000-of-00001.parquet'


dataset = load_dataset('parquet', data_files=dataset_path,split='train')
print(dataset)

# if model_name == 'llama3_8B':
#     dataset = dataset.map(add_prompt_text, load_from_cache_file=False)
# else:
#     dataset = dataset.map(add_prompt_chat,load_from_cache_file=False)
#
dataset = dataset.map(add_prompt_chat,load_from_cache_file=False)

# 查看处理后的数据集
# print(dataset[0])  # 打印第一个样本
# print(dataset)

# 加载模型
llm_init(model_name)
print('model loaded!')

result = []
for i in tqdm(range(len(dataset))):
    subject = dataset[i]['subject']
    response = llm_response(dataset[i]['messages'],model_name)
    answer = extract_answer(response,model_name)
    result.append([subject,
                   answer,
                   chr(dataset[i]['answer']+65)])

domain_acc = metrics.accuracy(result,model_name)
print(domain_acc)

