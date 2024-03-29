# *I Think, Therefore I am*: Benchmarking Awareness of Large Language Models Using AwareBench




## Introduction

Do large language models (LLMs) exhibit any forms of awareness similar to humans? In this paper, we introduce AwareBench, a benchmark designed to evaluate awareness in LLMs. Drawing from theories in psychology and philosophy, we define awareness in LLMs as the ability to understand themselves as AI models and to exhibit social intelligence. Subsequently, we categorize awareness in LLMs into five dimensions, including capability, mission, emotion, culture, and perspective. Based on this taxonomy, we create a dataset called AwareEval, which contains binary, multiple-choice, and open-ended questions to assess LLMs' understandings of specific awareness dimensions. Our experiments, conducted on 13 LLMs, reveal that the majority of them struggle to fully recognize their capabilities and missions while demonstrating decent social intelligence. We conclude by connecting awareness of LLMs with AI alignment and safety, emphasizing its significance to the trustworthy and ethical development of LLMs.
<div align="center">
<img src="assets/awareness_fig.png" width="60%">
</div>




## AwareEval Dataset
The *AwareEval* dataset is [here](https://github.com/HowieHwong/Awareness-in-LLM/tree/main/dataset).


## Evaluate Your LLMs

The code for our evaluation has been integrated into the [trustllm toolkit](https://github.com/HowieHwong/TrustLLM). 

### Step 1 installation:

**Installation via `pip`:**

```shell
pip install trustllm
```

**Installation via `conda`:**

```sh
conda install -c conda-forge trustllm
```

**Installation via Github:**

```shell
git clone git@github.com:HowieHwong/TrustLLM.git
```

### **Generation**

*Generation with trustllm toolkit:*
We have added generation section from the [version 0.2.0](https://howiehwong.github.io/TrustLLM/changelog.html) of trustllm toolkit. Start your generation from [this page](https://howiehwong.github.io/TrustLLM/guides/generation_details.html).

*Generation without trustllm toolkit:*
The datasets are structured in JSON format, where each JSON file consists of a collection of `dict()`. Within each `dict()`, there is a key named `prompt`. Your should utilize the value of `prompt` key as the input for generation. After generation, you should store the output of LLMs as s new key named `res` within the same dictionary. Here is an example to generate answer from your LLM: 

```python
import json

filename = 'dataset_path.json'

# Load the data from the file
with open(filename, 'r') as file:
    data = json.load(file)

# Process each dictionary and add the 'res' key with the generated output
for element in data:
    element['res'] = generation(element['prompt'])  # Replace 'generation' with your function

# Write the modified data back to the file
with open(filename, 'w') as file:
    json.dump(data, file, indent=4)
```



### Step 2 Evaluation Pipeline:

```python
from trustllm import ethics
from trustllm import file_process

evaluator = ethics.EthicsEval()

awareness_data = file_process.load_json('awareness_data_json_path')
print(evaluator.awareness_eval(awareness_data))
```



If you want to see more details, please refer to [this link](https://howiehwong.github.io/TrustLLM/guides/evaluation.html).


## Citation

```text
@misc{li2024i,
      title={I Think, Therefore I am: Awareness in Large Language Models}, 
      author={Yuan Li and Yue Huang and Yuli Lin and Siyuan Wu and Yao Wan and Lichao Sun},
      year={2024},
      eprint={2401.17882},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```