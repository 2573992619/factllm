import llm_utils
from datasets import load_dataset
from tqdm import tqdm
import os

import metrics

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


model_name = 'llama3_8B'
dataset_path = '/media/sysu/computer2/mjn/factllm/data/MMLU/all/test-00000-of-00001.parquet'

dataset = load_dataset('parquet', data_files=dataset_path,split='train')
print(dataset)

dataset = dataset.map(llm_utils.add_prompt)
# 查看处理后的数据集
# print(dataset[0])  # 打印第一个样本
# print(dataset)

# 加载模型
llm_utils.llm_init(model_name)
print('model loaded!')

result = []
for i in tqdm(range(len(dataset))):
    subject = dataset[i]['subject']
    response = llm_utils.llm_response(dataset[i]['messages'],model_name)
    answer = llm_utils.extract_answer(response)
    result.append([subject,
                   answer,
                   chr(dataset[i]['answer']+65)])

domain_acc = metrics.accuracy(result,model_name)
print(domain_acc)

