# Krutrim-2
[![Static Badge](https://img.shields.io/badge/Huggingface-Krutrim_2-yellow?logo=huggingface)](
https://huggingface.co/krutrim-ai-labs/Krutrim-2-instruct)	[![Static Badge](https://img.shields.io/badge/Github-Krutrim_2-green?logo=github)](https://github.com/ola-krutrim/Krutrim-2-12B)	[![Static Badge](https://img.shields.io/badge/Krutrim_Cloud-Krutrim_2-orange?logo=data:image/png%2bxml;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAADpUlEQVRYCbVXTUhbQRDeRJqi2JSS1BQtgpCa0kiPehdNi6dWbfWgF0s9eGtPFSFG8VDMpSiCB28KQou0NwsS9NA/Dx4qNP1TUIqSmlKSFjQx4vabbXbJz8vLe2kz8GX3zc7MN2/2J/sszLichekN4A7gBZxpcLQ/0gijfQq8BFLAf5ELiBIEfgNEZgSxtA/5liw2eD4EfgJGSLVsyPcBQLFMiR3WIUAraCm6F4hFMQ2JB1afgFKI9Hw+IubVYhnQwvpSBnKZ2GfEvlgoiTMYeFNGcpnEK3AQV548gkYalbslLiGWdEtl2QbOpZ9FMzg4yGprazNVpvrr6+tseXlZy+cXlFeAAzk4i07eW29sbPB/kampqbyYGTzEyagC5wHKJG+v6lWgqamJdXV1wY2xhYUFtr1NBcwWnQqQYRJwUQK3gOeArjidTkakJMfHx6y+vp4tLi6KZ5/Px1ZWVkTf5M9tstcsP/SifFarlQcCAX50dKRm4/T0lPf19ann9vZ2Xl1dzZubm3lVVZVe2XPHxDS8k2Ra7fj4uCKSnUgkwnt7e+Uj393d5ZQUSSqV4sFgMJeo0DNxsx0tYtLR2x8eHorA4XCY19TUqECZCZAB1gDf398XtvTT0dGhbAvFh37Hip9LgKbYbDZWWVkpxtbW1tjBgdo1rKGhQegTiQQbHR1lbreb9fT0qDgtLS2qr9MR3AkYFMyW3pwkGo3yzs5OPjAwwFdXV4WOfra2tpSv3W5X+snJSaXXiU/chaeAHLu7u1VQrQ6VXhJgWyqT/v5+pZfjGu0OdEx3EZJTW1sbX1pa4pgGgZmZGT40NCTIMisgDy5MC3c4HEYSEItwlkjMQi7Cvb095etyufjc3ByfmJhQuiJxiVscREYdlN3w8DA/OTnhsVhM6YqQadndpAToKNZdiLmBvV4vTyaTYgo2Nze5xWLRCl5MR0exOv5NTcPY2Jiaf2zTYkSFxkX56RwgCQBUBUNSUVEh7OicoP3e2trKpqenGf1fGBTi8ufaPoGiULZZ+sbGRh6Px9WWk52RkZEsO514j3PJ6Zlure8BQ0E8Hg+fn58X2zIUCnG/38/r6uqM+L4Fx9/jFZ1cuQzFN8BIoFJsviJ20Xm6DqN4GZKIIqYbMCQOWL0GSnlLLR+6rVBMU0I75B4QAbSCGtF9h+99QO42dM0L3ZRp1Zr9OCWfrFu2FrW8lmuN5erOQuED7gLXAPl5TjHk5/kH9J8BdBc39Hn+BxqB1clokCTRAAAAAElFTkSuQmCC)](https://cloud.olakrutrim.com/console/inference-service?section=models&modelName=Krutrim&artifactName=Krutrim-2&artifactType=model)	[![Static Badge](https://img.shields.io/badge/Krutrim_AI_Labs-Krutrim_2-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iMzYiIGhlaWdodD0iMzYiIHZpZXdCb3g9IjAgMCAzNiAzNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjM2IiBoZWlnaHQ9IjM2IiByeD0iMTgiIGZpbGw9IiMxMEE1NTQiLz4KPHBhdGggZD0iTTI2LjQxNCAxMi41OTE5SDE5LjMzVjE1LjY0OTlDMjAuMDM0IDE1LjIzOTIgMjAuODQwNyAxNS4wMzM5IDIxLjc1IDE1LjAzMzlDMjIuNzkxMyAxNS4wMzM5IDIzLjY0MiAxNS4zNTY1IDI0LjMwMiAxNi4wMDE5QzI0Ljk3NjcgMTYuNjQ3MiAyNS4zMTQgMTcuNTQxOSAyNS4zMTQgMTguNjg1OUMyNS4zMTQgMTkuMzMxMiAyNS4xODkzIDIwLjA0OTkgMjQuOTQgMjAuODQxOUMyNC43MDUzIDIxLjYzMzkgMjQuMzE2NyAyMi40NDA1IDIzLjc3NCAyMy4yNjE5TDIxLjIgMjEuODMxOUMyMS41MzczIDIxLjM3NzIgMjEuODE2IDIwLjkwNzkgMjIuMDM2IDIwLjQyMzlDMjIuMjU2IDE5LjkzOTkgMjIuMzY2IDE5LjQ0MTIgMjIuMzY2IDE4LjkyNzlDMjIuMzY2IDE4LjM4NTIgMjIuMjQ4NyAxOC4wMDM5IDIyLjAxNCAxNy43ODM5QzIxLjc5NCAxNy41NjM5IDIxLjUwMDcgMTcuNDUzOSAyMS4xMzQgMTcuNDUzOUMyMC43OTY3IDE3LjQ1MzkgMjAuMTQ0IDE3Ljc2MTkgMjAuMTQ0IDE3Ljc2MTlDMjAuMTQ0IDE3Ljc2MTkgMTkuMTE0NyAxOC4xODcyIDE4Ljg4IDE4LjQyMTlWMjMuODU1OUgxNi4zODJWMjEuMDYxOUMxNS44OTggMjEuMzQwNSAxNS40MDY3IDIxLjU1MzIgMTQuOTA4IDIxLjY5OTlDMTQuNDI0IDIxLjg0NjUgMTMuODU5MyAyMS45MTk5IDEzLjIxNCAyMS45MTk5QzEyLjQwNzMgMjEuOTE5OSAxMS42NjY3IDIxLjc3MzIgMTAuOTkyIDIxLjQ3OTlDMTAuMzMyIDIxLjE3MTkgOS44MDQgMjAuNzI0NSA5LjQwOCAyMC4xMzc5QzkuMDEyIDE5LjU1MTIgOC44MTQgMTguODE3OSA4LjgxNCAxNy45Mzc5QzguODE0IDE3LjExNjUgOS4wMTIgMTYuNDEyNSA5LjQwOCAxNS44MjU5QzkuODA0IDE1LjIyNDUgMTAuMzU0IDE0Ljc2MjUgMTEuMDU4IDE0LjQzOTlDMTEuNzYyIDE0LjEwMjUgMTIuNTc2IDEzLjkzMzkgMTMuNSAxMy45MzM5QzEzLjkxMDcgMTMuOTMzOSAxNC4zMjEzIDEzLjk0ODUgMTQuNzMyIDEzLjk3NzlDMTUuMTU3MyAxNC4wMDcyIDE1LjQ4NzMgMTQuMDU4NSAxNS43MjIgMTQuMTMxOUwxNS41MDIgMTYuNTczOUMxNS4wMzI3IDE2LjQ1NjUgMTQuNTEyIDE2LjM5NzkgMTMuOTQgMTYuMzk3OUMxMy4yNTA3IDE2LjM5NzkgMTIuNzE1MyAxNi41MzcyIDEyLjMzNCAxNi44MTU5QzExLjk1MjcgMTcuMDc5OSAxMS43NjIgMTcuNDUzOSAxMS43NjIgMTcuOTM3OUMxMS43NjIgMTguNTI0NSAxMS45NDUzIDE4LjkyNzkgMTIuMzEyIDE5LjE0NzlDMTIuNjc4NyAxOS4zNjc5IDEzLjA3NDcgMTkuNDc3OSAxMy41IDE5LjQ3NzlDMTQuMTE2IDE5LjQ3NzkgMTQuNjU4NyAxOS4zMzg1IDE1LjEyOCAxOS4wNTk5QzE1LjYxMiAxOC43ODEyIDE2LjAzIDE4LjQ1ODUgMTYuMzgyIDE4LjA5MTlWMTIuNTkxOUg4VjEwLjE3MTlIMjYuNDE0VjEyLjU5MTlaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMjIuMDc0IDI4Ljk4MTlDMjEuNjkyNyAyOS4xNzI1IDIxLjIzOCAyOS4zNDg1IDIwLjcxIDI5LjUwOTlDMjAuMTY3MyAyOS42NzEyIDE5LjUyMiAyOS43NTE5IDE4Ljc3NCAyOS43NTE5QzE4LjA0MDcgMjkuNzUxOSAxNy4zODggMjkuNjEyNSAxNi44MTYgMjkuMzMzOUMxNi4yNDQgMjkuMDY5OSAxNS43OTY3IDI4LjY5NTkgMTUuNDc0IDI4LjIxMTlDMTUuMTM2NyAyNy43NDI1IDE0Ljk2OCAyNy4xOTI1IDE0Ljk2OCAyNi41NjE5QzE0Ljk2OCAyNS41MDU5IDE1LjM0MiAyNC42NjI1IDE2LjA5IDI0LjAzMTlDMTYuODIzMyAyMy40MTU5IDE3LjQyOTMgMjMuMDYzOSAxOC44MDggMjIuOTc1OUwxOS4wNzIgMjUuMjQxOUMxOC4zMjQgMjUuMjg1OSAxOC4yNjA3IDI1LjQyNTIgMTcuOTgyIDI1LjY1OTlDMTcuNzAzMyAyNS45MDkyIDE3LjU2NCAyNi4xOTUyIDE3LjU2NCAyNi41MTc5QzE3LjU2NCAyNy4xOTI1IDE4LjAxMTMgMjcuNTI5OSAxOC45MDYgMjcuNTI5OUMxOS4yNDMzIDI3LjUyOTkgMTkuNTg4IDI3LjQ3ODUgMTkuOTQgMjcuMzc1OUMyMC4yOTIgMjcuMjczMiAyMC43MTczIDI3LjA5NzIgMjEuMjE2IDI2Ljg0NzlMMjIuMDc0IDI4Ljk4MTlaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K)](https://ai-labs.olakrutrim.com/models/Krutrim-LLM-2)

## Model Overview
Krutrim-2 is a 12B parameter language model developed by the OLA Krutrim team. It is built on the Mistral-NeMo 12B architecture and trained across various domains, including web data, code, math, Indic languages, Indian context data, synthetic data, and books. Following pretraining, the model was finetuned on diverse data covering a wide range of tasks, including knowledge recall, math, reasoning, coding, safety, instruction following and creative writing.

After fine-tuning, the model underwent Direct Preference Optimization (DPO) to enhance alignment across multiple aspects. DPO was applied to improve response helpfulness, safety, and reasoning.

The model delivers best-in-class performance across Indic tasks and a promising performance on English benchmarks equivalent to models 5-10x the size. We present details of the model architecture, pre-training, post-training and evaluation results. We also publicly release the post-trained versions of the model. We are continuously improving the model through post-training techniques such as RLHF. 

[![Krutrim 2](https://img.youtube.com/vi/beqXNHq67xg/0.jpg)](https://www.youtube.com/watch?v=beqXNHq67xg)


## Key Features
- 12B parameter dense transformer model leading to better generalization compared to Krutrim-1 7B;
- Supports context up to 128K tokens making it suitable for long multi-turn conversations, long-form generations, document translations and others;
- Delivers competitive performance on most English benchmarks and HumanEval coding task;
- Natively multilingual delivering best-in-class performance on Indic benchmarks;
- Matches or exceeds performance of models much larger (x6) on multilingual Indic generation tasks including creative writing, summarization, and translation;
- Stronger Indian cultural context relevance - scored the highest in manual evaluation with multiple models in an anonymised setting;
- Delivers top-3 performance on 5 (out of 7) tasks in BharatBench among much larger open source and commercial models. 
- Available in instruction-tuned versions;

## Model Developer
- OLA Krutrim Team

## Model Dates
- Krutrim-2 was trained between Dec 2024 and Jan 2025.

## Release History

| Model Name | Release Date |Release Note | Reference|
|------------|-------------|-------------|-------------|
| Krutrim-2-Instruct  | 2024-01-31 | Trained with MN-12B architecture |
| Krutrim-2-Instruct  | 2024-01-31 | Finetuned and aligned version of Krutrim-2-Base |[Here](https://huggingface.co/krutrim-ai-labs/Krutrim-2-instruct)


## Data Freshness
- The dataset includes information up to April 2024.

## Model Architecture
- Layers: 40
- Hidden Dimension: 5,120
- Head Dimension: 128
- Hidden Dimension: 14,336
- Activation Function: SiLU
- Number of Heads: 32
- Number of KV-Heads: 8 (GQA)
- Rotary Embeddings: Theta = 1M
- Vocabulary Size: 131072 (2<sup>17</sup>)
- Architecture Type: Transformer Decoder (Auto-regressive Language Model)

## Evaluation Results

### English/Code/Math Benchmarks


| Benchmark                                 | Krutrim-1-7B | MN-12B-Instruct| Krutrim-2-12B | llama-3.3-70B       | Gemini-1.5 Flash       | GPT-4o                 |
|-------------------------------------------|--------------|----------------|--------------------|----------------------|------------------------|-----------------------|
| Hellaswag (0-shot) - Accuracy             | 0.74         | 0.82           | 0.83               | 0.95                 | 0.87 (10-shot)         | 0.95 (10-shot)        |
| Winogrande (0-shot) - Accuracy            | 0.67         | 0.74           | 0.77               | 0.85 (5-shot)        | -                      | 0.88 (5-shot)         |
| OpenBookQA (0-shot) - Accuracy            | 0.45         | 0.46           | 0.49               | -                    | -                      | -                     |
| CommonSenseQA (0-shot) - Accuracy         | 0.74         | 0.70           | 0.74               | -                    | -                      | 0.85                  |
| TruthfulQA (0-shot) - Accuracy            | 0.49         | 0.54           | 0.59               | -                    | -                      | 0.59                  |
| MMLU (5-shot) - Accuracy                  | 0.47         | 0.68           | 0.63               | 0.82                 | 0.79                   | 0.86                  |
| TriviaQA (5-shot) - EM                    | 0.44         | 0.72           | 0.62               | -                    | -                      | -                     |
| NaturalQuestions (5-shot) - EM            | 0.15         | 0.28           | 0.26               | -                    | -                      | -                     |
| GSM8K (0-shot) - EM                       | 0.07         | 0.74           | 0.71               | 0.93 (8-shot, CoT)   | 0.86 (11-shot)         | 0.89                  |
| ARC_Challenge (0-shot) - Accuracy         | 0.48         | 0.59           | 0.60               | 0.93 (25-shot)       | -                      | 0.50                  |
| ARC_Easy (0-shot) - Accuracy              | 0.73         | 0.80           | 0.82               | -                    | -                      | -                     |
| HumanEval - Pass@10                       | 0.00         | 0.23           | 0.80               | 0.88                 | 0.74 (0-shot)          | 0.90                  |
| IF_Eval (0-shot) - Accuracy               | 0.27         | 0.46           | 0.73               | 0.92                 | -                      | 0.84                  |

### Indic Benchmarks

| Benchmark                                  | Metric     | Krutrim-1-7B | MN-12B-Instruct | Krutrim-2-12B | llama-3.1-8B | llama-3.3-70B | Gemini-1.5 Flash | GPT-4o |
|--------------------------------------------|------------|--------------|----------------|--------------|--------------|--------------|----------------|--------|
| IndicSentiment (0-shot)                   | Accuracy   | 0.65         | 0.70           | 0.95         | 0.05         | 0.96         | 0.99           | 0.98   |
| IndicCOPA (0-shot)                        | Accuracy   | 0.51         | 0.58           | 0.80         | 0.48         | 0.83         | 0.88           | 0.91   |
| IndicXParaphrase (0-shot)                 | Accuracy   | 0.67         | 0.74           | 0.88         | 0.75         | 0.87         | 0.89           | TBD    |
| IndicXNLI (0-shot)                        | Accuracy   | 0.47         | 0.54           | 0.55         | 0.00         | TBD          | TBD            | TBD    |
| IndicQA (0-shot)                          | Bert Score | 0.90         | 0.90           | 0.91         | TBD          | TBD          | TBD            | TBD    |
| CrossSumIN (1-shot)                       | chrF++     | 0.04         | 0.17           | 0.21         | 0.21         | 0.26         | 0.24           | TBD    |
| FloresIN Translation xx-en (1-shot)       | chrF++     | 0.54         | 0.50           | 0.58         | 0.54         | 0.60         | 0.62           | 0.63   |
| FloresIN Translation en-xx (1-shot)       | chrF++     | 0.41         | 0.34           | 0.48         | 0.37         | 0.46         | 0.47           | 0.48   |
| IN22 Translation xx-en (0-shot)           | chrF++     | 0.50         | 0.48           | 0.57         | 0.49         | 0.58         | 0.55           | TBD    |
| IN22 Translation en-xx (0-shot)           | chrF++     | 0.36         | 0.33           | 0.45         | 0.32         | 0.42         | 0.44           | TBD    |


### BharatBench
The existing Indic benchmarks are not natively in Indian languages, rather, they are translations of existing En benchmarks. They do not sufficiently capture the linguistic nuances of Indian languages and aspects of Indian culture. Towards that Krutrim released BharatBench - a natively Indic benchmark that encompasses the linguistic and cultural diversity of the Indic region, ensuring that the evaluations are relevant and representative of real-world use cases in India.

| Benchmark                           | Metric      | Krutrim-1-7B | MN-12B-Instruct | Krutrim-2-12B | llama-3.1-8B-Instruct | llama-3.1-70B-Instruct | Gemma-2-9B-Instruct | Gemma-2-27B-Instruct | GPT-4o |
|-------------------------------------|------------|--------------|-----------------|---------------|------------------------|------------------------|---------------------|---------------------|--------|
| Indian Cultural Context (0-shot)    | Bert Score | 0.86         | 0.56            | 0.88          | 0.87                   | 0.88                   | 0.87                | 0.87                | 0.89   |
| Grammar Correction (5-shot)         | Bert Score | 0.96         | 0.94            | 0.98          | 0.95                   | 0.98                   | 0.96                | 0.96                | 0.97   |
| Multi Turn (0-shot)                 | Bert Score | 0.88         | 0.87            | 0.91          | 0.88                   | 0.90                   | 0.89                | 0.89                | 0.92   |
| Multi Turn Comprehension (0-shot)   | Bert Score | 0.90         | 0.89            | 0.92          | 0.92                   | 0.93                   | 0.91                | 0.91                | 0.94   |
| Multi Turn Translation (0-shot)     | Bert Score | 0.85         | 0.87            | 0.92          | 0.89                   | 0.91                   | 0.90                | 0.91                | 0.92   |
| Text Classification (5-shot)        | Accuracy   | 0.61         | 0.71            | 0.76          | 0.72                   | 0.88                   | 0.82                | 0.86                | 0.89   |
| Named Entity Recognition (5-shot)   | Accuracy   | 0.31         | 0.51            | 0.53          | 0.55                   | 0.61                   | 0.61                | 0.65                | 0.65   |
### Qualitative Results
Below are the results from manual evaluation of prompt-response pairs across languages and task categories. Scores are between 1-5 (higher the better). Model names were anonymised during the evaluation.

![Cumulative Score - Category](./assets/CumulativeScoreCategory.png)
![Cumulative Score - Language](./assets/CumulativeScoreLanguage.png)

## Usage

To run this model, do this:
```
git clone https://github.com/ola-krutrim/Krutrim-2-12B.git
cd Krutrim-2-12B
pip install -r requirements.txt
```

To use the instruct model, you can load it with `AutoModelForCausalLM` as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "krutrim-ai-labs/Krutrim-2-instruct"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

To generate, first format the prompt in OpenAI Chat Message Format and apply chat template. 
```python

prompt_dict = [
    {"role":'system','content':"You are an AI assistant."},
    {"role":'user','content':"Who are you?"}
]
prompt = tokenizer.apply_chat_template(prompt_dict, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(prompt, return_tensors='pt')
inputs.pop("token_type_ids", None)

# Generate response
outputs = model.generate(
    **inputs,
    max_length=100
)

response = tokenizer.decode(outputs[0])
```

To test batch inference of instruct model, you can run
```
python inference/batch_inference.py
```

## Limitations
The model was trained on a dataset that includes content from the internet, which may contain toxic language, biases, and unsafe content. As a result, the model may:
- Amplify biases present in the training data
- Generate toxic responses, especially when prompted with toxic inputs
- Provide inaccurate, incomplete, or redundant answers
- Generate responses in languages inconsistent with the prompt

## License
[Krutrim Community License](./LICENSE)

## Ethical Considerations
- The model may produce biased or offensive outputs based on its training data.
- Users should apply human oversight when using the model for decision-making in sensitive areas.
- While safeguards have been implemented, the model may still generate socially undesirable text in certain contexts.

## Contact
TBD

