import math
from huggingface_hub import notebook_login

notebook_login()

# For each of those tasks, we will use the Wikitext 2 dataset as an example. 
# You can load it very easily with the 🤗 Datasets library.
from datasets import load_dataset

# We will use the gpt2 architecture for this example. 
# You can pick any of the checkpoints listed here instead. 
# For the tokenizer, you can replace the checkpoint by the one you trained yourself.
model_checkpoint = "gpt2"
tokenizer_checkpoint = "NewPlus/my-new-shiny-tokenizer"

# To tokenize all our texts with the same vocabulary that was used when training the model, we have to download a pretrained tokenizer. 
# This is all done by the AutoTokenizer class:
from transformers import AutoTokenizer    
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# We can now call the tokenizer on all our texts. 
# This is very simple, using the map method from the Datasets library. 
# First we define a function that call the tokenizer on our texts:
def tokenize_function(examples):
    return tokenizer(examples["text"])

def train():
    # Then we apply it to all the splits in our datasets object, using batched=True and 4 processes to speed up the preprocessing. 
    # We won't need the text column afterward, so we discard it.
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # If we now look at an element of our datasets, we will see the text have been replaced by the input_ids the model will need:
    print(tokenized_datasets["train"][1])

    # Now for the harder part: we need to concatenate all our texts together then split the result in small chunks of a certain block_size. 
    # To do this, we will use the map method again, with the option batched=True. 
    # This option actually lets us change the number of examples in the datasets by returning a different number of examples than we got. 
    # This way, we can create our new samples from a batch of examples.
    # First, we grab the maximum length our model was pretrained with. This might be a big too big to fit in your GPU RAM, so here we take a bit less at just 128.

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

    # First note that we duplicate the inputs for our labels. This is because the model of the 🤗 Transformers library apply the shifting to the right, 
    # so we don't need to do it manually.
    # Also note that by default, the map method will send a batch of 1,000 examples to be treated by the preprocessing function. 
    # So here, we will drop the remainder to make the concatenated tokenized texts a multiple of block_size every 1,000 examples. 
    # You can adjust this behavior by passing a higher batch size (which will also be processed slower). 
    # You can also speed-up the preprocessing by using multiprocessing:
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    # And we can check our datasets have changed: now the samples contain chunks of block_size contiguous tokens, potentially spanning over several of our original texts.
    print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))

    # Now that the data has been cleaned, we're ready to instantiate our Trainer. First we create the model using the same config as our checkpoint, but initialized with random weights:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_config(config)

    # And we will needsome TrainingArguments:
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        f"{model_checkpoint}-wikitext2",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=True,
    )

    # The last argument to setup everything so we can push the model to the Hub regularly during training. 
    # Remove it if you didn't follow the installation steps at the top of the notebook. 
    # If you want to save your model locally in a name that is different than the name of the repository it will be pushed, 
    # or if you want to push your model under an organization and not your name space, use the hub_model_id argument to set the repo name 
    # (it needs to be the full name, including your namespace: 
    # for instance "sgugger/gpt-finetuned-wikitext2" or "huggingface/gpt-finetuned-wikitext2").
    # We pass along all of those to the Trainer class:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    # And we can train our model:
    print(trainer.train())

    # The perplexity is still quite high since for this demo we trained on a small dataset for a small number of epochs. 
    # For a real LM training, you would need a larger dataset and more epochs.
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.push_to_hub()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("NewPlus/my-awesome-model")

if __name__ == '__main__':
    train()