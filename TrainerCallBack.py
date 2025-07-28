import torch
import numpy as np
import pandas as pd
import os
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm

# --- Step 1: Define Your Helper Functions ---

def calculate_text_overlap(pred_text, true_text):
    """Calculates a simple word-level Jaccard similarity."""
    pred_tokens = set(pred_text.lower().split())
    true_tokens = set(true_text.lower().split())
    if not true_tokens and not pred_tokens: return 1.0
    if not true_tokens or not pred_tokens: return 0.0
    intersection = len(pred_tokens.intersection(true_tokens))
    union = len(pred_tokens.union(true_tokens))
    return intersection / union

def generate_detailed_report(evaluation_results):
    """Creates a detailed report from evaluation results and returns a pandas DataFrame."""
    print("Generating detailed report DataFrame...")
    report_df = pd.DataFrame(evaluation_results)
    return report_df

def get_smarter_answer(question, context, model, tokenizer):
    """Your custom inference logic to get a single predicted answer string."""
    device = model.device
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=384, truncation="only_second").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
    start_index = torch.argmax(start_logits).item()
    search_window = end_logits[start_index : start_index + 50]
    end_index_relative = torch.argmax(search_window).item()
    end_index = start_index + end_index_relative
    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer


# --- Step 2: Create a Custom Trainer by Subclassing ---
# This is the new, correct approach that replaces the callback.

class CustomTrainer(Trainer):
    """
    A custom trainer that overrides the evaluation loop to add a custom metric
    and generate a detailed report.
    """
    def __init__(self, *args, eval_dataset_raw=None, inference_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the raw (untokenized) evaluation dataset and inference function
        self.eval_dataset_raw = eval_dataset_raw
        self.inference_function = inference_function

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Part 1: Run the default evaluation (computes loss)
        # The super().evaluate() call returns a dictionary of the default metrics
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # Part 2: Run our custom evaluation logic
        print("\n--- Running Custom Overlap Evaluation ---")
        
        evaluation_results = []
        for example in tqdm(self.eval_dataset_raw, desc="Custom Eval"):
            question = example['question']
            context = example['context']
            true_answer = example['answers']['text'][0] if example['answers']['text'] else ""
            predicted_answer = self.inference_function(question, context, self.model, self.tokenizer)
            score = calculate_text_overlap(predicted_answer, true_answer)
            
            evaluation_results.append({
                'question': question,
                'context': context,
                'predicted_answer': predicted_answer,
                'true_answer': true_answer,
                'overlap_score': score
            })
            
        # Part 3: Calculate our custom metric and add it to the metrics dictionary
        all_scores = [result['overlap_score'] for result in evaluation_results]
        avg_score = np.mean(all_scores)
        
        # Add the metric to the dictionary with the correct prefix (e.g., "eval_custom_overlap")
        metrics[f"{metric_key_prefix}_custom_overlap"] = avg_score
        
        print(f"Custom Overlap Score: {avg_score:.4f}")

        # Part 4: Generate and save the detailed report
        report_df = generate_detailed_report(evaluation_results)
        epoch = int(self.state.epoch) if self.state.epoch is not None else "final"
        report_path = os.path.join(self.args.output_dir, f"evaluation_report_epoch_{epoch}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"Saved detailed evaluation report to: {report_path}")
        print("--- Finished Custom Evaluation ---")

        # Part 5: Log all metrics (default + custom) to history and print to console
        self.log(metrics)

        return metrics


# --- Step 3: Putting It All Together in Your Training Script ---

# Assume you have: model, tokenizer, tokenized_train_dataset, 
# tokenized_val_dataset, and the original untokenized val_dataset

# 1. Configure TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    
    # This will now work correctly because our custom metric is part of the main evaluation output
    load_best_model_at_end=True,
    metric_for_best_model="custom_overlap", 
    greater_is_better=True, 
    
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
)

# 2. Initialize our CustomTrainer instead of the default Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    # Pass our custom arguments here
    eval_dataset_raw=val_dataset,
    inference_function=get_smarter_answer
)

# 3. Train the model
trainer.train()

# After training, trainer.model will be the one from the checkpoint
# that had the highest 'custom_overlap' score.
