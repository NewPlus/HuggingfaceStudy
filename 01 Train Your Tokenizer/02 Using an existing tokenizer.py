from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

from transformers import AutoTokenizer

# If you want to train a tokenizer with the exact same algorithms and parameters as an existing one, you can just use the train_new_from_iterator API.
# For instance, let's train a new version of the GPT-2 tokenzier on Wikitext-2 using the same tokenization algorithm.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Make sure that the tokenizer you picked as a fast version (backed by the 🤗 Tokenizers library) otherwise the rest of the notebook will not run:
print(tokenizer.is_fast)
# Then we feed the training corpus (either the list of list or the iterator we defined earlier)
# to the train_new_from_iterator method. We also have to specify the vocabulary size we want to use:
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)
print(new_tokenizer(dataset[:5]["text"]))
new_tokenizer.save_pretrained("my-new-tokenizer")

from huggingface_hub import notebook_login

# Huggingface Login with token
# Need **Write Token**
# notebook_login()
# Cloning https://huggingface.co/NewPlus/my-new-shiny-tokenizer into local empty directory.
# new_tokenizer.push_to_hub("my-new-shiny-tokenizer")

# Or from anywhere using the repo ID, which is your namespace followed by a slash an the name you gave in the push_to_hub method, so for instance:
tok = new_tokenizer.from_pretrained("NewPlus/my-new-shiny-tokenizer")

# Now if you want to create and a train a new tokenizer that doesn't look like anything in existence, you will need to build it from scratch using the 🤗 Tokenizers library.