import math
from huggingface_hub import notebook_login
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

notebook_login()

# For each of those tasks, we will use the Wikitext 2 dataset as an example. 
# You can load it very easily with the 🤗 Datasets library.
from datasets import load_dataset

model_checkpoint = "bert-base-cased"
tokenizer_checkpoint = "sgugger/bert-like-tokenizer"
    
# We can apply the same tokenization function as before, we just need to update our tokenizer to use the checkpoint we just picked:
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

def tokenize_function(examples):
    return tokenizer(examples["text"])

# For masked language modeling (MLM) we are going to use the same preprocessing as before for our dataset with one additional step: 
# we will randomly mask some tokens (by replacing them by [MASK]) 
# and the labels will be adjusted to only include the masked tokens (we don't have to predict the non-masked tokens). 
# If you use a tokenizer you trained yourself, make sure the [MASK] token is among the special tokens you passed during training!
# We will use the bert-base-cased model for this example. 
# You can pick any of the checkpoints listed here instead. For the tokenizer, replace the checkpoint by the one you trained.

# block_size = tokenizer.model_max_length
block_size = 128

# Then we write the preprocessing function that will group our texts:
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train():
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    from transformers import AutoConfig, AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_config(config)

    training_args = TrainingArguments(
        "test-clm",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True,
        push_to_hub_model_id=f"{model_checkpoint}-wikitext2",
    )

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == '__main__':
    train()