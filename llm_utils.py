import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def llm_init(model_name):
    global device
    global model
    global tokenizer
    global pipeline

    # 对于不同模型选择相应的加载方式
    if model_name == "mistral-8B":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("/media/sysu/computer2/mjn/models/Ministral-8B-Instruct", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("/media/sysu/computer2/mjn/models/Ministral-8B-Instruct")
        model.config.pad_token_id = tokenizer.eos_token_id  # 使用 eos_token 作为 pad_token
        # print(model)
        model.to(device)

    if model_name == "mistral-7B":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("/media/sysu/computer2/mjn/models/Mistral-7B", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("/media/sysu/computer2/mjn/models/Mistral-7B")
        model.config.pad_token_id = tokenizer.eos_token_id  # 使用 eos_token 作为 pad_token

        model.to(device)

    if model_name == "Qwen-7B-omni":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("/media/sysu/computer2/mjn/models/Qwen-7B-omni", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("/media/sysu/computer2/mjn/models/Qwen-7B-omni")
        model.to(device)

    if model_name == "llama3_8B":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("/media/sysu/computer2/mjn/models/Llama3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("/media/sysu/computer2/mjn/models/Llama3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct")
        model.config.pad_token_id = tokenizer.eos_token_id  # 使用 eos_token 作为 pad_token
        model.to(device)

    if model_name == "gemma3-12B":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("/media/sysu/computer2/mjn/models/google-gemma-3-12b-it", device_map="auto",
                                                     torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("/media/sysu/computer2/mjn/models/google-gemma-3-12b-it")

    if model_name == "DeepSeek-R1-7B":
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("/media/sysu/computer2/mjn/models/DeepSeek-R1-7B",
                                                     torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("/media/sysu/computer2/mjn/models/DeepSeek-R1-7B")
        model.to(device)

    # if model_name == "chatgpt":
    #     openai.api_key = os.getenv("OPENAI_API_KEY")


def llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200):
    #针对不同模型，取得推理结果
    if model_name == "mistral-8B":
        message = prompt
        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=3, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)
        print('model output decoded', decoded)

        # answer = extract_answer(decoded[0])
        # print(answer)
        # print('model answer output:\n', decoded[0])
        return decoded[0]

    if model_name == "mistral":

        message = prompt
        encodeds = tokenizer.apply_chat_template(message,
                                                 return_tensors="pt",
                                                 add_generation_prompt=True)
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs,
                                       max_new_tokens=1,
                                       do_sample=False,
                                       min_new_tokens=1
                                       )
        decoded = tokenizer.batch_decode(generated_ids)
        # print('model output decoded', decoded)
        # answer = extract_math_expression(decoded[0])
        print('model answer output:\n', decoded[0])

        return decoded[0]

    if model_name == "llama3_8B":
        message = prompt
        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=10, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)
        print('model output decoded', decoded)
        # answer = extract_math_expression(decoded[0])
        # print('model answer output\n', decoded[0])
        return decoded[0]

    if model_name == "DeepSeek-R1-7B":
        message = prompt
        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")
        model_inputs = encodeds.to(device)
        model.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=50, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids)
        print('model output decoded', decoded)
        # answer = extract_math_expression(decoded[0])
        print('model answer output\n', decoded[0])
        return decoded[0]

def add_prompt_chat(examples):

    e1 = """Answer the following questions by selecting only the final answer: A, B, C, or D. 
Statement 1 | If H is a subgroup of a group G and a belongs to G, then aH = Ha. Statement 2 | If H is normal of G and a belongs to G, then ah = ha for all h in H.
A: True, True
B: False, False
C: True, False
D: False, True"""
    a1 = "B"


    problem_msg = 'Answer the following questions by selecting only the final answer: A, B, C, or D. \n' + "Problem:\n" + examples['question'] + "\n"
    for i, choice in enumerate(examples["choices"]):
        problem_msg += f"{chr(65 + i)}: {choice}\n"

    messages = [
        {"role": "user", "content": e1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": problem_msg}
    ]
    # return {"input": messages}
    # print('messages')
    return {
        "messages": messages
    }

def add_prompt_text(examples):
    prompt = """**Choice Problem Solver**
    Task Instruction:
    Answer the following questions by selecting only the final answer: A, B, C, or D. 

    Format Requirements:
    1. No explanatory text - only output the choice

    Examples:
    USER: "Statement 1 | If H is a subgroup of a group G and a belongs to G, then aH = Ha. Statement 2 | If H is normal of G and a belongs to G, then ah = ha for all h in H.
    A: True, True
    B: False, False
    C: True, False
    D: False, True"
    ASSISTANT: B

    USER: "Based on the characteristic population curves that result from plotting population growth of a species, the most effective means of controlling the mosquito population is to
    A:maintain the population at a point corresponding to the midpoint of its logistic curve
    B:opt for zero population control once the K value of the curve has been reached
    C:reduce the carrying capacity cif the environment to lower the K value
    D:increase the mortality rate"
    ASSISTANT: C

    Problem:"""
    problem_msg = prompt + examples['question'] + "\n"
    for i, choice in enumerate(examples["choices"]):
        problem_msg += f"{chr(65 + i)}: {choice}\n"

    return {
        "messages": problem_msg
    }


def extract_answer(text,model_name):
    '''
    从模型输出中提取具体答案选项,模型生成的答案不符合预期时返回NAN
    '''
    if model_name == 'llama3_8B':
        segments = text.split("\n")
        # print(segments)
        for text in reversed(segments):
            if text.strip() != '':
                answer_text = segments[-1]
                break

        try:
            if answer_text[0] == 'A' or 'B' or 'C' or 'D':
                answer = answer_text[0]
                print("提取的答案:", answer)
                return answer

        except:

            match = re.search(r"([ABCDabcd])<\|eot_id\|>", answer_text)

            if match:
                answer = match.group(1).upper()  # 提取并统一为大写
                print("提取的答案:", answer)
                return answer
            else:
                print("没有匹配内容#######################")
                return 'NAN'


        # for segment in reversed(segments):
        #     if '<|eot_id|>' in segment:
        #         answer = segment[0]
        #         print('提取的answer:',answer)
        #         if answer in ['A', 'B', 'C', 'D']:
        #             return answer
        #         if answer in ['a', 'b', 'c', 'd']:
        #             return answer.upper()
        #         print('提取失败#################################')
        #         return 'NAN'
        # print('提取失败#################################')
        # return 'NAN'

    if model_name == 'mistral':
        segments = text.split("\n")
        answer = segments[-1]
        if answer in ['A', 'B', 'C', 'D']:
            return answer
        if answer in ['a', 'b', 'c', 'd']:
            return answer.upper()
        return 'NAN'

    if model_name == 'mistral-8B':
        segments = text.split("\n")
        answe_text = segments[-1]
        match = re.search(r"\[/INST\]\s*([ABCDabcd])", answe_text)

        if match:
            answer = match.group(1).upper()  # 如需要统一为大写
            print("提取的答案:", answer)
            return answer
        else:
            print("没有匹配内容")
            return 'NAN'
        return 'NAN'


        # if answer in ['A', 'B', 'C', 'D']:
        #     return answer
        # if answer in ['a', 'b', 'c', 'd']:
        #     return answer.upper()
        # return 'NAN'