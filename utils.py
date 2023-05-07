from datasets import load_dataset

def format(inp):
    PROMPT_FORMAT = '以下は、あるタスクを記述した指示です。依頼を適切に完了させる回答を書きなさい。\n\n### 指示:\n{instruction}\n\n### 応答:\n'
    try:
        if inp['input'] != '':
            instruction = inp['instruction'] + '\n' + inp['input']
        else:
            instruction = inp['instruction']
        prompt = PROMPT_FORMAT.format(instruction=instruction)
        response = inp['output']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return {'prompt': prompt, 'response': response}    


def prepare_dataset(tokenizer):    
    ds_path = 'https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja/resolve/main/databricks-dolly-15k-ja.json'
    ds = load_dataset('json', data_files=ds_path)
    print(ds['train'][0])

    def tokenize_example(example):
        example = format(example)
        tokenizer(text=example['prompt'], text_target=example['response'])
        return example
    columns_to_remove = list(ds[0].keys())
    ds = ds.map(
        tokenize_example, 
        batched=True,
        remove_columns=columns_to_remove
    )
    
    # print("data_set:", len(ds))    
    # print("train_data:", len(train_data))
    # print("val_data:", len(val_data))
    return ds