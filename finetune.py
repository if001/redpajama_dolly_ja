from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM

from composer import Trainer, algorithms
from composer.core import Evaluator
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import LinearWithWarmupScheduler

MIN_TRANSFORMERS_VERSION = '4.25.1'
# check transformers version
import transformers
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'


from transformers import (
    HfArgumentParser,    
    DataCollatorForSeq2Seq
)

from utils import (
    prepare_dataset,
    get_device
)

# max_seq_length = 256
# max_dataset_length = 80000

# max_seq_length = 256
# max_dataset_length = 200


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "model name or model path"}
    )
    output_model_path: str = field(default=None)

    def __post_init__(self):
        debug_print(self)


def debug_print(s):
    # GREEN = '\033[32m'
    print('\033[32m'+str(s)+'\033[0m')

def load_model(model_name):
    model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map='auto', 
                                                 torch_dtype=torch.float16, 
                                                 load_in_8bit=True)
    print("load model:", model_name)
    return tokenizer, model


def arg_parse() -> Tuple[ModelArguments]:
    parser = HfArgumentParser((ModelArguments))
    model_args = parser.parse_args_into_dataclasses()    
    return model_args

def main():
    model_args = arg_parse()

    tokenizer, model = load_model(model_args.model_name)

    ds = prepare_dataset(tokenizer)

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_seq_len=1024,
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
                            metric_names=list(model.train_metrics.keys()))
    
    model.to(get_device())
    print('model is cuda', model.device)
    optimizer = DecoupledAdamW(model.parameters(),
                              lr=1.0e-5,
                              betas=(0.9, 0.999),
                              eps=1.0e-8,
                              weight_decay=0)
    scheduler = LinearWithWarmupScheduler(t_warmup='0ba', alpha_f=0)
    al = algorithms.GradientClipping({
        'clipping_type': 'norm',
        'clipping_threshold': 1.0
    })
    trainer = Trainer(
        run_name='redpajama_dolly_ja',
        model=model,
        train_dataloader=train_dataloader,
        eval_loader=[eval_loader],
        tokenizer=tokenizer,
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=al,
        max_duration='1ep',
        eval_interval=10,
        log_to_console=True,
        save_folder=model_args.output_model_path,
        save_filename=''
    )
    print("train...")
    # trainer.train(resume)
    trainer.fit()
    print("evaluate...")

if __name__ == "__main__":
    main()