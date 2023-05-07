import argparse

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    EarlyStoppingCallback
)

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
                                                load_in_8bit=True,
                                                 device_map='auto', 
                                                 torch_dtype=torch.float16)                                                 
    print("load model:", model_name)
    return tokenizer, model

def with_lora(model):
    from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
    print('load with lora...')

    model = prepare_model_for_int8_training(model, 
                                            use_gradient_checkpointing=True)    
    LORA_R=16
    LORA_ALPHA=32
    LORA_DROPOUT=0.05
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=False,
        target_modules = ["query_key_value"],
    )
    model = get_peft_model(model, config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='/content/MyDrive/models/redpajama_dolly_ja')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_ac', default=8, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lora', action='store_true')
    args = parser.parse_args()


    model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    tokenizer, model = load_model(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if args.lora:
        model = with_lora(model)

    train_dataset = prepare_dataset(tokenizer)
    
    # model.to(get_device())
    # print('model is cuda', model.device)

    
    training_args = TrainingArguments(
        evaluation_strategy="step",
        eval_steps=100,        
        save_strategy='step',
        save_steps=5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        output_dir=args.out_dir,
        gradient_accumulation_steps=args.grad_ac,
        lr_scheduler_type='constant',
        learning_rate=1e-5,
        metric_for_best_model = 'eval_loss',
        load_best_model_at_end = True,
        save_total_limit=3,
        fp16 = True,
        gradient_checkpointing= True,        
        warmup_steps=100
        )
    # optim="adafactor", for row gpu vram
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
        )
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator        
        )
    print("train...")
    trainer.train(args.resume)
    
    if args.lora:
        model.save_pretrained("alpaca-lora-dolly-2.0")
    else:
        print("evaluate...")
        trainer.evaluate()
        trainer.save_model()

if __name__ == "__main__":
    main()