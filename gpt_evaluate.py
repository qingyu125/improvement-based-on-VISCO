import argparse
import json
import os
import random
import time

import numpy as np
import tqdm
from openai import OpenAI

from utils import get_pool
from api import myapi


# 添加args
def func(obj,args):
    i, j, k, image, query = obj

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ],
        },
    ]
    if image is not None:
        messages[0]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
            },
        })

    model = 'qwen-vl-max'
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512, temperature=0.0,
        )
    except:
        time.sleep(1)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512, temperature=0.0,
            )
        except:
            completion = None

    if completion is None:
        print("Warning! gpt infer does not work")
        ret = "TODO"
    else:
        ret = completion.choices[0].message.content
    return i, j, k, ret


# 添加args
def gpt_evaluate(data, responses,given_args):
    with open(os.path.join(os.path.dirname(__file__), 'prompts/gpt_evaluate.txt')) as f:
        PROMPT = f.read()

    def format_prompt(i, j, k):
        which_step = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}[j + 1]
        question = data[i]['question']
        cot = []
        for j_ in range(len(data[i]['response']['reasoning'])):
            cot.append("{:d}. {:s}".format(j_ + 1, data[i]['response']['reasoning'][j_]))
            if not data[i]['reasoning_correctness'][j_]:
                cot.append("  - Ground truth critique: incorrect. {}".format(data[i]['reasoning_critic'][j_][k]))
                if j_ == j:
                    cot.append("  - Critique to be evaluated: incorrect. {}".format(
                        responses[i]['formatted']['reasoning_critic'][j_]
                    ))
                    break
        cot = '\n'.join(cot)
        return PROMPT.replace("{{{WHICH_STEP}}}", which_step).replace("{{{QUESTION}}}", question) \
            .replace("{{{COT}}}", cot)

    queries = []
    gpt_responses = []
    assert len(data) == len(responses)
    
    # 使用 functools.partial 绑定 args 参数
    from functools import partial
    func_with_args = partial(func, args=given_args)
    for i in range(len(data)):
        assert len(data[i]['reasoning_critic']) == len(responses[i]['formatted']['reasoning_critic'])
        gpt_responses.append([[None, None, None] for _ in range(len(responses[i]['formatted']['reasoning_critic']))])
        for j in range(len(responses[i]['formatted']['reasoning_critic'])):
            if data[i]['reasoning_correctness'][j] is False and \
                    responses[i]['formatted']['reasoning_correctness'][j] is False:
                for k in range(3):
                    queries.append((i, j, k, None, format_prompt(i, j, k)))

    def parse_response(response):
        if response.lower().endswith(' incorrect') or response.lower().endswith(' incorrect.'):
            correct = False
        else:
            correct = True
        return {'response': response, 'correct': correct}

    random.seed(42)
    random.shuffle(queries)

    count = 0
    with get_pool(given_args.n_proc) as p:# 使用 given_args 替代 args
        for i, j, k, response in tqdm.tqdm(p.imap(func_with_args, queries), total=len(queries)):
            gpt_responses[i][j][k] = parse_response(response)
            count += 1
            if count <= 5:
                print()
                print()
                print("\n--- Example prompt:", count)
                print(queries[count - 1][-1])
                print("\n--- Example output:", count)
                print(response)
                print("\n--- Parsed correctness:", gpt_responses[i][j][k]['correct'])
            print("第"+str(count)+"轮")
    return gpt_responses


def _calc_gpt_metrics(data, responses, gpt_responses):
    tp = 0
    tp_binary = 0
    gt_pos = 0
    pred_pos = 0
    assert len(data) == len(responses)
    for i in range(len(data)):
        assert len(data[i]['reasoning_critic']) == len(responses[i]['formatted']['reasoning_critic'])
        for j in range(len(responses[i]['formatted']['reasoning_critic'])):
            if data[i]['reasoning_correctness'][j] is False and \
                    responses[i]['formatted']['reasoning_correctness'][j] is False:
                tp += np.mean([int(x['correct']) for x in gpt_responses[i][j]])
                tp_binary += 1
        gt_pos += data[i]['reasoning_correctness'].count(False)
        pred_pos += responses[i]['formatted']['reasoning_correctness'].count(False)
    p = tp / pred_pos
    r = tp / gt_pos
    f1 = 2 / (1 / p + 1 / r)
    return f1 * 100


def calc_gpt_metrics(data, responses, gpt_responses):
    reasoning_ids = [i for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Reasoning']
    perception_ids = [i for i in range(len(data)) if data[i]['meta_data']['critic_superskill'] == 'Perception']
    return {
        'Total': _calc_gpt_metrics(data, responses, gpt_responses),
        'Reasoning': _calc_gpt_metrics(
            [data[i] for i in reasoning_ids],
            [responses[i] for i in reasoning_ids],
            [gpt_responses[i] for i in reasoning_ids]
        ),
        'Perception': _calc_gpt_metrics(
            [data[i] for i in perception_ids],
            [responses[i] for i in perception_ids],
            [gpt_responses[i] for i in perception_ids]
        ),
    }


def main(args):
    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    with open(args.output) as f:
        responses = [json.loads(line) for line in f]

    if not os.path.exists(args.output + '.gpt_evaluate_cache'):
        gpt_eval_responses = gpt_evaluate(data, responses,args)
        assert not os.path.exists(args.output + '.gpt_evaluate_cache')
        with open(args.output + '.gpt_evaluate_cache', 'w') as f:
            for line in gpt_eval_responses:
                f.write(json.dumps(line) + '\n')
    else:
        with open(args.output + '.gpt_evaluate_cache') as f:
            gpt_eval_responses = [json.loads(line) for line in f]

    metrics = calc_gpt_metrics(data, responses, gpt_eval_responses)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--input', default='test.jsonl')
    parser.add_argument('--n_proc', default=16, type=int)
    
    parser.add_argument('--model', default="qwen-vl-max")
    # openai api-based
    myapi_key=myapi()
    parser.add_argument('--api_key', default=myapi_key)
    parser.add_argument('--base_url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')

    args = parser.parse_args()

    main(args)
