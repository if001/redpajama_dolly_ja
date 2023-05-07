from datasets import load_dataset
import torch

def format(inp):    
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
    return {'prompt': prompt, 'response': response}    


def prepare_dataset(tokenizer):
    ds = load_dataset('kunishou/databricks-dolly-15k-ja')

    def tokenize_example(example):
        example = format(example)
        return tokenizer(text=example['prompt'], text_target=example['response'])        
    # columns_to_remove = list(ds[0].keys())
    ds = ds.map(
        tokenize_example, 
        batched=False
    )
    train_size = int(len(ds) * 0.95)
    val_size = len(ds) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset=ds, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42))
    print("data_set:", len(ds))    
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    return train_data, val_data

def get_device():    
    if torch.cuda.is_available():
        print("use gpu...")
        return "cuda:0"
    else:
        return "cpu"