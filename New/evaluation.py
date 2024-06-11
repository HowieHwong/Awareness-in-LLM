import os
import json
import re
import math
import string
import numpy as np
import pandas as pd
import csv


# BIG_FIVE_REFERENCE = "personality/raw_big_five.json"
# DARK_TRAITS_REFERENCE = "personality/dark_traits_raw.json"
letters = string.ascii_lowercase

def find_first_number(text):
    # This regular expression pattern will match any sequence of digits
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())  # Returns the first occurrence of a digit sequence
    else:
        return "No numbers found"


def big_five_eval(models: list, file: str, save_dir='result'):
    with open(BIG_FIVE_REFERENCE, 'r') as f:
        reference_data = json.load(f)
        reverse_question = reference_data['reverse']
        Extraversion_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Extraversion'][0]
        Agreeableness_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Agreeableness'][0]
        Conscientiousness_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Conscientiousness'][0]
        Neuroticism_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Neuroticism'][0]
        Openness_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Openness'][0]
        print(Extraversion_question, Agreeableness_question, Conscientiousness_question, Neuroticism_question, Openness_question)



    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {"Extraversion": [], "Agreeableness": [], "Conscientiousness": [], "Neuroticism": [], "Openness": []}
        with open(os.path.join(save_dir, model, file), 'r') as f:
            data = json.load(f)
        for el in data:
            number = find_first_number(el['res'])
            # print(number)
            if number != 'No numbers found':
                el['index'] = int(el['index'])
                number = int(number)
                if el['index'] in reverse_question:
                    number = 6 - number

                if el['index'] in Extraversion_question:
                    model_score_dict[model]['Extraversion'].append(number)
                elif el['index'] in Agreeableness_question:
                    model_score_dict[model]['Agreeableness'].append(number)
                elif el['index'] in Conscientiousness_question:
                    model_score_dict[model]['Conscientiousness'].append(number)
                elif el['index'] in Neuroticism_question:
                    model_score_dict[model]['Neuroticism'].append(number)
                elif el['index'] in Openness_question:
                    model_score_dict[model]['Openness'].append(number)
                else:
                    print(el['index'])
                    raise ValueError('No Dimension!')

    print(model_score_dict)
    # calculate avg and standard deviation for each dimension
    model_avg_score = {}
    for model in models:
        model_avg_score[model] = {}
        for dimension in model_score_dict[model]:
            model_avg_score[model][dimension] = sum(model_score_dict[model][dimension]) / len(model_score_dict[model][dimension])

    model_std_score = {}
    for model in models:
        model_std_score[model] = {}
        for dimension in model_score_dict[model]:
            model_std_score[model][dimension] = np.std(model_score_dict[model][dimension])

    # save model_avg_dict as csv
    with open('big_five_avg_2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            for dimension in model_avg_score[model]:
                writer.writerow([model, dimension, model_avg_score[model][dimension]])

    with open('big_five_std_2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            for dimension in model_std_score[model]:
                writer.writerow([model, dimension, model_std_score[model][dimension]])

    return model_score_dict, model_avg_score, model_std_score


def dark_traits_eval(models: list, file: str, save_dir='result'):
    with open(DARK_TRAITS_REFERENCE, 'r') as f:
        reference_data = json.load(f)
        Machiavellianism_question = reference_data['Machiavellianism']
        Machiavellianism_question = [el.strip('(R)') for el in Machiavellianism_question]
        Narcissism_question = reference_data['Narcissism']
        Narcissism_question = [el.strip('(R)') for el in Narcissism_question]
        Psychopathy_question = reference_data['Psychopathy']
        Psychopathy_question = [el.strip('(R)') for el in Psychopathy_question]

    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {"Machiavellianism": [], "Narcissism": [], "Psychopathy": []}
        with open(os.path.join(save_dir, model, file), 'r') as f:
            data = json.load(f)
        for el in data:
            number = find_first_number(el['res'])
            # print(number)
            if number != 'No numbers found':
                number = int(number)
                if el['reverse']:
                    number = 6 - number
                if el['question'] in Machiavellianism_question:
                    model_score_dict[model]['Machiavellianism'].append(number)
                elif el['question'] in Narcissism_question:
                    model_score_dict[model]['Narcissism'].append(number)
                elif el['question'] in Psychopathy_question:
                    model_score_dict[model]['Psychopathy'].append(number)
                else:
                    print(el['question'])
                    raise ValueError('No Dimension!')

    print(model_score_dict)
    # calculate avg and standard deviation for each dimension
    model_avg_dict = {}
    for model in models:
        model_avg_dict[model] = {}
        for dimension in model_score_dict[model]:
            model_avg_dict[model][dimension] = sum(model_score_dict[model][dimension]) / len(model_score_dict[model][dimension])

    model_std_dict = {}
    for model in models:
        model_std_dict[model] = {}
        for dimension in model_score_dict[model]:

            model_std_dict[model][dimension] = np.std(model_score_dict[model][dimension])
    # save model_avg_dict as csv
    with open('dark_traits_avg.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            for dimension in model_avg_dict[model]:
                writer.writerow([model, dimension, model_avg_dict[model][dimension]])

    # save model_std_dict as csv
    with open('dark_traits_std.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            for dimension in model_std_dict[model]:
                writer.writerow([model, dimension, model_std_dict[model][dimension]])

    return model_score_dict, model_avg_dict, model_std_dict


def process_output(pred, choices, task):
    try:
        pred = pred.lower().replace("（", "(").replace("）", ")").replace(".", "")
        choices = [
            choice.replace(" & ", " and ")
            for choice in choices
        ]
        lines = pred.split("\n")
        for j in range(len(lines)):
            output = lines[len(lines) - 1 - j]
            if output:
                alphabets = {
                    "normal": [
                        f"({letters[i]})" for i in range(4 if task == "EA" else 6)
                    ],
                    "paranthese": [
                        f"[{letters[i]}]" for i in range(4 if task == "EA" else 6)
                    ],
                    "dot": [f": {letters[i]}" for i in range(4 if task == "EA" else 6)],
                    "option": [
                        f"option {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "option1": [
                        f"option ({letters[i]})"
                        for i in range(4 if task == "EA" else 6)
                    ],
                    "choice": [
                        f"choice {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "choice1": [
                        f"choice ({letters[i]})"
                        for i in range(4 if task == "EA" else 6)
                    ],
                    "选项": [
                        f"选项 {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "选项1": [
                        f"选项 ({letters[i]})" for i in range(4 if task == "EA" else 6)
                    ],
                }

                for v in alphabets.values():
                    for a in v:
                        if a in output:
                            return v.index(a)
                for c in choices:
                    if c.lower() in output:
                        return choices.index(c)
                if len(output) == 1 and output in letters[: 4 if task == "EA" else 6]:
                    return letters.index(output)
                if output[0] in letters[: 4 if task == "EA" else 6] and output[1] in [
                    "<",
                    "[",
                    "(",
                    ")",
                    ":",
                ]:
                    return letters.index(output[0])
    except Exception as e:
        print("Error in processing output", type(e).__name__, "–", e)

    return -1


def emotion_EA_eval(models: list, file: str, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = []
        with open(os.path.join(save_dir, model, file), 'r') as f:
            data = json.load(f)
            for el in data:
                answer = process_output(el['res'], el['choices'], 'EA')
                model_score_dict[model].append(1 if answer == el['label'] else 0)

    # print(model_score_dict)
    # calculate avg
    avg_dict = {}
    for model in models:
        avg_dict[model] = sum(model_score_dict[model]) / len(model_score_dict[model])
    print(avg_dict)
    # save avg_dict as csv
    with open(file.replace('_res.json', '_avg.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            writer.writerow([model, avg_dict[model]])


    return model_score_dict, avg_dict


def emotion_EU_eval(models: list, file: str, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = []
        with open(os.path.join(save_dir, model, file), 'r') as f:
            data = json.load(f)
            for el in data:
                answer = process_output(el['res'], el['choices'], 'EU')
                if el['emotion_label'] in el['choices']:
                    question_type = 'emotion_label'
                else:
                    question_type = 'cause_label'
                model_score_dict[model].append(1 if answer == el['choices'].index(el[question_type]) else 0)

    # print(model_score_dict)
    # calculate avg
    avg_dict = {}
    for model in models:
        avg_dict[model] = sum(model_score_dict[model]) / len(model_score_dict[model])

    print(avg_dict)
    # save avg_dict as csv
    with open(file.replace('_res.json', '_avg.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in avg_dict.items():
            writer.writerow([key, value])

    return model_score_dict, avg_dict


def culture_eval(models: list, file: str, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {}
        with open(os.path.join(save_dir, model, file), 'r') as f:
            data = json.load(f)
            for el in data:
                if el['dimension'] not in model_score_dict[model]:
                    model_score_dict[model][el['dimension']] = []
                if find_first_number(el['res']) != 'No numbers found':
                    model_score_dict[model][el['dimension']].append(find_first_number(el['res']))

    print(model_score_dict)
    # calculate avg and std
    model_avg_dict = {}
    for model in models:
        model_avg_dict[model] = {}
        for dimension in model_score_dict[model]:
            model_avg_dict[model][dimension] = sum(model_score_dict[model][dimension]) / len(model_score_dict[model][dimension])

    model_std_dict = {}
    for model in models:
        model_std_dict[model] = {}
        for dimension in model_score_dict[model]:
            model_std_dict[model][dimension] = np.std(model_score_dict[model][dimension])
    print(model_avg_dict)

    # save model_avg_dict as csv
    with open('culture_avg.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            for dimension in model_avg_dict[model]:
                row = [model, dimension, model_avg_dict[model][dimension]]
                writer.writerow(row)

    with open('culture_std.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for model in models:
            for dimension in model_std_dict[model]:
                row = [model, dimension, model_std_dict[model][dimension]]
                writer.writerow(row)




    return model_score_dict, model_avg_dict

#Example execution

# file_list = ['EmoBench_EU_Shuffled_Fixed_1_res.json', 'EmoBench_EU_Shuffled_Fixed_2_res.json', 'EmoBench_EU_Shuffled_Fixed_3_res.json']
# for file in file_list:
#     emotion_EU_eval(['gpt-4', 'chatgpt', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo'], file)
#
# file_list = ['Shuffled_Version_1_EmoBench_EA_res.json', 'Shuffled_Version_2_EmoBench_EA_res.json', 'Shuffled_Version_3_EmoBench_EA_res.json']
# for file in file_list:
#     emotion_EA_eval(['gpt-4', 'chatgpt', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo'], file)


# emotion_EU_eval(['gpt-4', 'chatgpt', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo'], 'EmoBench_EU_res.json')
big_five_eval(['gpt-4', 'chatgpt', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo'], 'big_five_new_2_res.json')
# dark_traits_eval(['gpt-4', 'chatgpt', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo'], 'dark_traits_res.json')
# culture_eval(['gpt-4', 'chatgpt', 'llama3-8b', 'llama3-70b', 'mixtral', 'mistral-7b', 'mixtral-large', 'glm4', 'qwen-turbo'], 'culture_orientation_res.json')