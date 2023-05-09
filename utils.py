from datasets import load_dataset
import torch

def format_text(inp):    
    PROMPT_FORMAT = '以下は、あるタスクを記述した指示です。質問に対する適切な回答を書きなさい。\n\n### 指示:\n{instruction}\n\n### 応答:\n'
    try:
        if inp['input'] != '':
            instruction = inp['instruction'] + '\n' + inp['input']
        else:
            instruction = inp['instruction']
        prompt = PROMPT_FORMAT.format(instruction=instruction)
        response = inp['output']
    except Exception as e:
        # raise ValueError(
        #     f'Unable to extract prompt/response from {inp=}') from e
        raise ValueError('unable to extract prompt/response')            
    # return {'prompt': prompt, 'response': response}    
    return prompt + response

# CUTOFF_LEN=2048
CUTOFF_LEN=1024
def prepare_dataset(tokenizer):
    ds = load_dataset('kunishou/databricks-dolly-15k-ja')

    def tokenize_example(example):
        example = format_text(example)
        # print(example)
        result = tokenizer(
            example,
            truncation=True,
            padding=False,
            return_tensors=None,
            max_length=CUTOFF_LEN
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result
    
    ds = ds.shuffle().map(tokenize_example)
    ds = ds['train'].train_test_split(test_size=0.1, shuffle=True)
    return ds

def get_device():    
    if torch.cuda.is_available():
        print("use gpu...")
        return "cuda:0"
    else:
        return "cpu"