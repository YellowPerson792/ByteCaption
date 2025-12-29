---
library_name: vllm
language:
- en
- fr
- es
- de
- it
- pt
- nl
- zh
- ja
- ko
- ar
license: apache-2.0
inference: false
extra_gated_description: >-
  If you want to learn more about how we process your personal data, please read
  our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
tags:
- mistral-common
---

# Ministral 3 8B Base 2512
A balanced model in the Ministral 3 family, **Ministral 3 8B** is a powerful, efficient tiny language model with vision capabilities.

This model is the base pre-trained version, not fine-tuned for instruction or reasoning tasks, making it ideal for custom post-training processes.  
For instruction and chat based use cases, we recommend using [Ministral 3 8B Instruct 2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512).

The Ministral 3 family is designed for edge deployment, capable of running on a wide range of hardware. Ministral 3 8B can even be deployed locally, capable of fitting in 24GB of VRAM in BF16, and less than 12GB of RAM/VRAM when quantized.

## Key Features
Ministral 3 8B consists of two main architectural components:
- **8.4B Language Model**
- **0.4B Vision Encoder**

The Ministral 3 8B Base model offers the following capabilities:
- **Vision**: Enables the model to analyze images and provide insights based on visual content, in addition to text.
- **Multilingual**: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic.
- **Edge-Optimized**: Delivers best-in-class performance at a small scale, deployable anywhere.
- **Apache 2.0 License**: Open-source license allowing usage and modification for both commercial and non-commercial purposes.
- **Large Context Window**: Supports a 256k context window.

### Use Cases
Perfect for balanced performance in local or embedded systems, combining versatility with efficiency.
- Chat interfaces in constrained environments
- Local daily-driver AI assistant
- Image/document description and understanding
- Translation and content generation
- Specialized agentic use cases
- Fine-tuning and specialization
- And more...
  
Bringing advanced AI capabilities to resource-constrained environments.

## Ministral 3 Family

| Model Name                     | Type               | Precision | Link                                                                                     |
|--------------------------------|--------------------|-----------|------------------------------------------------------------------------------------------|
| Ministral 3 3B Base 2512       | Base pre-trained   | BF16      | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)                |
| Ministral 3 3B Instruct 2512   | Instruct post-trained | FP8   | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)            |
| Ministral 3 3B Reasoning 2512  | Reasoning capable  | BF16      | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)           |
| **Ministral 3 8B Base 2512**       | **Base pre-trained**   | **BF16**      | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512)                |
| Ministral 3 8B Instruct 2512   | Instruct post-trained | FP8    | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)            |
| Ministral 3 8B Reasoning 2512  | Reasoning capable  | BF16      | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512)           |
| Ministral 3 14B Base 2512      | Base pre-trained   | BF16      | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-14B-Base-2512)               |
| Ministral 3 14B Instruct 2512  | Instruct post-trained | FP8    | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512)           |
| Ministral 3 14B Reasoning 2512 | Reasoning capable  | BF16      | [Hugging Face](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512)          |

Other formats available [here](https://huggingface.co/collections/mistralai/ministral-3-additional-checkpoints).

## Benchmark Results

We compare Ministral 3 to similar sized models.

### Reasoning

| Model                     | AIME25      | AIME24      | GPQA Diamond | LiveCodeBench |
|---------------------------|-------------|-------------|--------------|---------------|
| **Ministral 3 14B**       | <u>0.850</u>| <u>0.898</u>| <u>0.712</u> | <u>0.646</u>  |
| Qwen3-14B (Thinking)      | 0.737       | 0.837       | 0.663        | 0.593         |
|                           |             |             |              |               |
| **Ministral 3 8B**        | 0.787       | <u>0.860</u>| 0.668        | <u>0.616</u>  |
| Qwen3-VL-8B-Thinking      | <u>0.798</u>| <u>0.860</u>| <u>0.671</u> | 0.580         |
|                           |             |             |              |               |
| **Ministral 3 3B**        | <u>0.721</u>| <u>0.775</u>| 0.534        | <u>0.548</u>  |
| Qwen3-VL-4B-Thinking      | 0.697       | 0.729       | <u>0.601</u> | 0.513         |

### Instruct

| Model                     | Arena Hard  | WildBench  | MATH Maj@1  | MM MTBench       |
|---------------------------|-------------|------------|-------------|------------------|
| **Ministral 3 14B**       | <u>0.551</u>| <u>68.5</u>| <u>0.904</u>| <u>8.49</u>      |
| Qwen3 14B (Non-Thinking)  | 0.427       | 65.1       | 0.870       | NOT MULTIMODAL   |
| Gemma3-12B-Instruct       | 0.436       | 63.2       | 0.854       | 6.70             |
|                           |             |            |             |                  |
| **Ministral 3 8B**        | 0.509       | <u>66.8</u>| 0.876       | <u>8.08</u>      |
| Qwen3-VL-8B-Instruct      | <u>0.528</u>| 66.3       | <u>0.946</u>| 8.00             |
|                           |             |            |             |                  |
| **Ministral 3 3B**        | 0.305       | <u>56.8</u>| 0.830       | 7.83             |
| Qwen3-VL-4B-Instruct      | <u>0.438</u>| <u>56.8</u>| <u>0.900</u>| <u>8.01</u>      |
| Qwen3-VL-2B-Instruct      | 0.163       | 42.2       | 0.786       | 6.36             |
| Gemma3-4B-Instruct        | 0.318       | 49.1       | 0.759       | 5.23             |

### Base

| Model               | Multilingual MMLU | MATH CoT 2-Shot | AGIEval 5-shot | MMLU Redux 5-shot | MMLU 5-shot | TriviaQA 5-shot |
|---------------------|-------------------|-----------------|----------------|-------------------|-------------|-----------------|
| **Ministral 3 14B** | 0.742             | <u>0.676</u>    | 0.648          | 0.820             | 0.794       | 0.749           |
| Qwen3 14B Base      | <u>0.754</u>      | 0.620           | <u>0.661</u>   | <u>0.837</u>      | <u>0.804</u>| 0.703           |
| Gemma 3 12B Base    | 0.690             | 0.487           | 0.587          | 0.766             | 0.745       | <u>0.788</u>    |
|                     |                   |                 |                |                   |             |                 |
| **Ministral 3 8B**  | <u>0.706</u>      | <u>0.626</u>    | 0.591          | 0.793             | <u>0.761</u>| <u>0.681</u>    |
| Qwen 3 8B Base      | 0.700             | 0.576           | <u>0.596</u>   | <u>0.794</u>      | 0.760       | 0.639           |
|                     |                   |                 |                |                   |             |                 |
| **Ministral 3 3B**  | 0.652             | <u>0.601</u>    | 0.511          | 0.735             | 0.707       | 0.592           |
| Qwen 3 4B Base      | <u>0.677</u>      | 0.405           | <u>0.570</u>   | <u>0.759</u>      | <u>0.713</u>| 0.530           |
| Gemma 3 4B Base     | 0.516             | 0.294           | 0.430          | 0.626             | 0.589       | <u>0.640</u>    |

## Usage

The model can be used with the following frameworks;
- [`vllm`](https://github.com/vllm-project/vllm): See [here](#vllm)
- [`transformers`](https://github.com/huggingface/transformers): See [here](#transformers)
  
### vLLM

We recommend using this model with [vLLM](https://github.com/vllm-project/vllm).

#### Installation

Make sure to install **vllm >= 1.12.0**:

```
pip install vllm --upgrade
```

Doing so should automatically install [`mistral_common >= 1.8.6`](https://github.com/mistralai/mistral-common/releases/tag/v1.8.6).

To check:
```
python -c "import mistral_common; print(mistral_common.__version__)"
```

You can also make use of a ready-to-go [docker image](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) or on the [docker hub](https://hub.docker.com/layers/vllm/vllm-openai/latest).

#### Serve

Due to their size and the BF16 format of their weights `Ministral-3-3B-Base-2512` and `Ministral-3-8B-Base-2512` can run on a single 1xH200 GPU.

A simple launch command is:

```bash
vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral
```

Additional flags:

* You can set `--max-model-len` to preserve memory. By default it is set to `262144` which is quite large but not necessary for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency.

#### Usage of the model

Here we asumme that the model `mistralai/Ministral-3-8B-Base-2512` is served and you can ping it to the domain `localhost` with the port `8000` which is the default for vLLM.

<details>
  <summary>Test Base</summary>

Quick test with the base model.

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 256

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

response = client.completions.create(
    model=model,
    prompt="What is the best thing in the universe ?",
    temperature=TEMP,
    max_tokens=MAX_TOK,
)

print(response.choices[0].text)
```

</details>

### Transformers

You can also use Ministral 3 8B Base 2512 with `Transformers` !
Make sure to install `Transformers` from its first v5 release candidate or from "main":

```
pip install transformers==5.0.0rc0
```

To make the best use of our model with `Transformers` make sure to have [installed](https://github.com/mistralai/mistral-common) `mistral-common >= 1.8.6` to use our tokenizer.

```bash
pip install mistral-common --upgrade
```

Then load our tokenizer along with the model and generate:

<details>
  <summary>Python snippet</summary>

```python
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config

model_id = "mistralai/Ministral-3-8B-Base-2512"
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
)
tokenizer = MistralCommonBackend.from_pretrained(model_id)

input_ids = tokenizer.encode("Once about a time, France was a", return_tensors="pt")
input_ids = input_ids.to("cuda")

output = model.generate(
    input_ids,
    max_new_tokens=30,
)[0]

decoded_output = tokenizer.decode(output[len(input_ids[0]):])
print(decoded_output)
```

</details>

## License

This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

*You must not use this model in a manner that infringes, misappropriates, or otherwise violates any third party’s rights, including intellectual property rights.*