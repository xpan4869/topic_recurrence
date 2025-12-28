import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

model_id = "meta-llama/Llama-3.3-70B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)

one_to_five_idxs = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

with open('/project/bermanm/datasets/political_word_embeddings/corpora/reddit_raw_comments_20250616/subsample.json', 'r', encoding='utf-8') as f:
    for _, line in enumerate(f):
        year_month = line_dict['year_month']
        keyword = line_dict['keyword']
        idx = line_dict['idx']
        if (year_month, idx) in processed:
            continue

        subreddit = line_dict['subreddit']
        body = line_dict['body']

        messages = [
        {"role": "system", "content": "You are a social psychology research assistant trained to rate the valence toward certain topics that the text is showing. Valence is the pleasantness or unpleasantness of an emotional stimulus, in this case expressed through text. Use a 1-5 Likert scale where 1 = very unpleasant, 2 = unpleasant, 3 = neutral, 4 = pleasant, 5 = very pleasant.\nConsider the valence toward the topic, not just the valence of the overall text.\nProvide only the score."},
        {"role": "user", "content": f"What is the valence toward {keyword} in the following text?\n{body}"},
        ]

        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        try:
            output = quantized_model(inputs, output_scores=True)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory. Skipping this example. {year_month}, {keyword}, {idx}, {subreddit}")
            torch.cuda.empty_cache()
            continue
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0).float().cpu().detach().numpy()
        likerts = next_token_probs[one_to_five_idxs]
        weighted_likert = sum([(i+1) * (likerts[i] / sum(likerts)) for i in range(5)])
        max_likert = np.argmax(likerts) + 1

        with open('./outputs/llama_likert.csv', 'a') as f:
            f.write(f'{year_month},{keyword},{idx},{subreddit},{",".join(likerts.astype(str))},{weighted_likert},{max_likert}\n')