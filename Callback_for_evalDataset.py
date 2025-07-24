import torch
import numpy as np
import pandas as pd
import os
from transformers import Trainer, TrainingArguments, TrainerCallback
from tqdm.auto import tqdm

# --- Step 1: Define Your Custom Metric Function ---
def calculate_text_overlap(pred_text, true_text):
    """Calculates a simple word-level Jaccard similarity."""
    pred_tokens = set(pred_text.lower().split())
    true_tokens = set(true_text.lower().split())
    if not true_tokens and not pred_tokens: return 1.0
    if not true_tokens or not pred_tokens: return 0.0
    intersection = len(pred_tokens.intersection(true_tokens))
    union = len(pred_tokens.union(true_tokens))
    return intersection / union

# --- NEW: Define Your Report Generation Function ---
def generate_detailed_report(evaluation_results):
    """
    Creates a detailed report from evaluation results.
    This function takes a list of dictionaries and returns a pandas DataFrame.
    You can add more complex logic here (e.g., calculating more metrics).
    """
    print("Generating detailed report DataFrame...")
    report_df = pd.DataFrame(evaluation_results)
    # You could add more columns or calculations here if needed
    # For example: report_df['is_exact_match'] = report_df['predicted_answer'] == report_df['true_answer']
    return report_df


# --- Step 2: Create the (Updated) Custom Callback ---
class CustomEvaluationCallback(TrainerCallback):
    """
    A custom callback to evaluate the model and generate a detailed report
    at the end of each evaluation phase.
    """
    def __init__(self, eval_dataset, tokenizer, inference_function):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.inference_function = inference_function

    def on_evaluate(self, args, state, control, model, **kwargs):
        """This event is triggered after the default evaluation is computed."""
        print("\n--- Running Custom Overlap Evaluation ---")
        
        evaluation_results = []
        
        for example in tqdm(self.eval_dataset, desc="Custom Eval"):
            question = example['question']
            context = example['context']
            true_answer = example['answers']['text'][0] if example['answers']['text'] else ""
            predicted_answer = self.inference_function(question, context, model, self.tokenizer)
            score = calculate_text_overlap(predicted_answer, true_answer)
            
            # Store all details needed for the report
            evaluation_results.append({
                'question': question,
                'context': context,
                'predicted_answer': predicted_answer,
                'true_answer': true_answer,
                'overlap_score': score
            })
            
        # --- Part 1: Calculate and log the primary metric for the Trainer ---
        all_scores = [result['overlap_score'] for result in evaluation_results]
        avg_score = np.mean(all_scores)
        state.log_history[-1]['eval_custom_overlap'] = avg_score
        print(f"Custom Overlap Score: {avg_score:.4f}")
        
        # --- Part 2: Generate and save the detailed CSV report ---
        report_df = generate_detailed_report(evaluation_results)
        
        # Create a unique filename for each epoch's report
        # state.epoch is a float, so we cast to int
        epoch = int(state.epoch) if state.epoch is not None else "final"
        report_path = os.path.join(args.output_dir, f"evaluation_report_epoch_{epoch}.csv")
        
        report_df.to_csv(report_path, index=False)
        print(f"Saved detailed evaluation report to: {report_path}")
        print("--- Finished Custom Evaluation ---")


# --- Step 3: Define Your Custom Inference Function ---
def get_smarter_answer(question, context, model, tokenizer):
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


# --- Step 4: Putting It All Together in Your Training Script ---
# Assume you have: model, tokenizer, train_dataset, val_dataset

# 1. Instantiate your callback
custom_eval_callback = CustomEvaluationCallback(
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    inference_function=get_smarter_answer
)

# 2. Configure TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="custom_overlap", 
    greater_is_better=True, 
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
)

# 3. Initialize the Trainer with the callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    callbacks=[custom_eval_callback],
)

# 4. Train the model
trainer.train()
