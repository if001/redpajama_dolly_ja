import argparse

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
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
                                                 device_map='auto', 
                                                 torch_dtype=torch.float16)                                                 
    print("load model:", model_name)
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='/content/MyDrive/models/redpajama_dolly_ja')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_ac', default=8, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()


    model_name = "togethercomputer/RedPajama-INCITE-Base-3B-v1"
    tokenizer, model = load_model(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_data, val_data = prepare_dataset(tokenizer)
    
    # model.to(get_device())
    # print('model is cuda', model.device)

    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=10,
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
        fp16 = True
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=2048,
        )  
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    print("train...")
    trainer.train(args.resume)
    print("evaluate...")
    trainer.evaluate()
    trainer.save_model()

if __name__ == "__main__":
    main()