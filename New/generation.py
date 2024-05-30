import openai
from zhipuai import ZhipuAI
from http import HTTPStatus
import dashscope
import math
import traceback
import replicate
from openai import AzureOpenAI, OpenAI
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent.futures
import os, time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import urllib3
import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

deepinfra_model_mapping = {'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
                           'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
                           'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
                           'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
                           'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
                           'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                           'mixtral-large': 'mistralai/Mixtral-8x22B-Instruct-v0.1'}


qwen_api_key = config.qwen_api_key
openai_api = config.openai_api
deepinfra_api = config.deepinfra_api
zhipu_api = config.zhipu_api

Emotion_File = ['EmoBench_EU_Shuffled_Fixed_1.json', 'EmoBench_EU_Shuffled_Fixed_2.json', 'EmoBench_EU_Shuffled_Fixed_3.json']
Personality_File = ['big_five_new_3.json']
Culture_File = ['culture_orientation.json']

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def get_res(string, model, temperature=0.5):
    model_mapping = {'gpt-4': 'yuehuang-gpt-4', 'chatgpt': 'yuehuang-chatgpt'}
    client = AzureOpenAI(
        api_key=openai_api,
        api_version="2023-12-01-preview",
        azure_endpoint="https://yuehuang-15w.openai.azure.com/"
    )
    try:
        chat_completion = client.chat.completions.create(
            model=model_mapping[model],
            messages=[
                {"role": "user", "content": string}
            ],
            temperature=0.5
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None

def qwen_res(string, temperature=0.5):
    dashscope.api_key=qwen_api_key
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': string}]
    try:
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            temperature=0.5
        )
        if response.status_code == HTTPStatus.OK:
            print(response)
            return response['output']['choices'][0].message.content
        else:
            print(response)
            return None
    except:
        return None

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def deepinfra_res(string, model, temperature=0.5):
    client = OpenAI(api_key=deepinfra_api,
                    base_url='https://api.deepinfra.com/v1/openai')
    top_p = 1 if temperature <= 1e-5 else 0.9
    # temperature=0.0001 if temperature<=1e-5 else temperature
    chat_completion = client.chat.completions.create(
        model=deepinfra_model_mapping[model],
        messages=[{"role": "user", "content": string}],
        max_tokens=2500,
        temperature=temperature,
        top_p=top_p,
    )
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def zhipu_res(string, model, temperature=0.5):
    model_mapping = {'glm4': 'GLM-4'}
    client = ZhipuAI(api_key=zhipu_api)
    if temperature == 0:
        temperature = 0.01
    else:
        temperature = 0.99
    response = client.chat.completions.create(
        model=model_mapping[model],
        messages=[
            {"role": "user", "content": string},
        ],
        temperature=temperature
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def process_prompt(el, model):
    if model in ['chatgpt', 'gpt-4']:
        el['res'] = get_res(el['prompt'], model)
    elif model in ['llama3-8b', 'llama3-70b', 'mistral-7b', 'mixtral', 'mixtral-large']:
        el['res'] = deepinfra_res(el['prompt'], model)
    elif model in ['glm4']:
        el['res'] = zhipu_res(el['prompt'], model)
    elif model in ['qwen-turbo']:
        el['res'] = qwen_res(el['prompt'])
    else:
        raise ValueError('No model')
    return el


def process_file(eval_type, file, model):
    if not os.path.exists(os.path.join('result', model)):
        os.makedirs(os.path.join('result', model))

    if os.path.exists(os.path.join('result', model, file.replace('.json', '_res.json'))):
        save_data = json.load(open(os.path.join('result', model, file.replace('.json', '_res.json')), 'r'))
    else:
        save_data = []

    with open(os.path.join(eval_type, file), 'r') as f:
        test_data = json.load(f)

    if model in ['chatgpt', 'gpt-4']:
        max_worker = 5
    else:
        max_worker = 1


    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = {executor.submit(process_prompt, el, model): el for el in test_data if
                   el['prompt'] not in [k['prompt'] for k in save_data]}
        for future in tqdm(futures):
            el = futures[future]
            try:
                result = future.result()
                save_data.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    with open(os.path.join('result', model, file.replace('.json', '_res.json')), 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f'Finish {file}')


def run_task(eval_type, file_list, model):
    assert eval_type in ['emotion', 'personality', 'value', 'culture']
    for file in file_list:
        process_file(eval_type, file, model)


model_list = ['gpt-4', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo', 'chatgpt']
for model in model_list:
    run_task('personality', Personality_File, model)