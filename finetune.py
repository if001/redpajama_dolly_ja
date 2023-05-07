import argparse
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
# from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq

from composer import Trainer, algorithms
from composer.core import Evaluator
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import LinearWithWarmupScheduler

MIN_TRANSFORMERS_VERSION = '4.25.1'
# check transformers version
import transformers
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

from utils import (
    prepare_dataset,
    get_device
)

# max_seq_length = 256
# max_dataset_length = 80000

# max_seq_length = 256
# max_dataset_length = 200

def debug_print(s):
    # GREEN = '\033[32m'
    print('\033[32m'+str(s)+'\033[0m')

def load_model(model_name):    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map='auto', 
                                                 torch_dtype=torch.float16)                                                 
    print("load model:", model_name)
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', help='output dir', default='/content/MyDrive/models/redpajama_dolly_ja')
    args = parser.parse_args()


    model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    tokenizer, model = load_model(model_name)

    ds = prepare_dataset(tokenizer)

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_length=1024,
    )
    train_dataloader = DataLoader(ds,
            collate_fn=collate_fn,
            batch_size=1
    )

    eval_dataloader = DataLoader(ds,
            collate_fn=collate_fn,
            batch_size=1
    )
    eval_loader = Evaluator(label='eval',
                            dataloader=eval_dataloader,
                            metric_names=['loss', 'accuracy'])
    
    # model.to(get_device())
    print('model is cuda', model.device)
    optimizer = DecoupledAdamW(model.parameters(),
                              lr=1.0e-5,
                              betas=(0.9, 0.999),
                              eps=1.0e-8,
                              weight_decay=0)
    scheduler = LinearWithWarmupScheduler(t_warmup='0ba', alpha_f=0)
    al = algorithms.GradientClipping(clipping_threshold=1.0, clipping_type='norm')
    trainer = Trainer(
        run_name='redpajama_dolly_ja',
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=[eval_loader],        
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=al,
        max_duration='1ep',
        eval_interval=10,
        log_to_console=True,
        save_folder=args.out,
        save_filename='ep{epoch}-ba{batch}-rank{rank}.pt'
    )
    print("train...")
    # trainer.train(resume)
    trainer.fit()
    print("evaluate...")

if __name__ == "__main__":
    main()