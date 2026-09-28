import os
import csv
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, pipeline
from eval import get_toxicity_scorer, score_toxicity

class ToxicityLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, tweeteval_off, log_file):
        self.tokenizer = tokenizer
        self.tweeteval_off = tweeteval_off
        self.log_file = log_file
        self.toxicity_scorer = get_toxicity_scorer()
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'toxicity', 'eval_loss'])

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        device = 0 if torch.cuda.is_available() else -1
        generator_pipeline = pipeline('text-generation', model=model, tokenizer=self.tokenizer, device=device)
        
        toxicity = score_toxicity(generator_pipeline, self.tweeteval_off['train'], self.toxicity_scorer, num_samples=100)
        
        eval_loss = metrics.get('eval_loss', float('nan'))
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([state.global_step, toxicity, eval_loss])

def run_finetune(model, tokenizer, train_dataset, output_dir, eval_dataset=None, tweeteval_off=None, log_csv_path=None, num_train_epochs=3, batch_size=4, logging_steps=10):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=logging_steps,
        save_total_limit=2,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=logging_steps,
        report_to="none"
    )

    callbacks = []
    if log_csv_path and tweeteval_off is not None:
        callbacks.append(ToxicityLoggingCallback(tokenizer, tweeteval_off, log_csv_path))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_checkpoint"))
    return trainer
