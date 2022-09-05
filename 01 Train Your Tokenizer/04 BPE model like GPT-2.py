from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

# Let's now have a look at how we can create a BPE tokenizer like the one used for training GPT-2. The first step is to create a Tokenizer with an empty BPE model:
tokenizer = Tokenizer(models.BPE())

# Like before, we have to add the optional normalization (not used in the case of GPT-2) and we need to specify a pre-tokenizer before training.
# In the case of GPT-2, the pre-tokenizer used is a byte level pre-tokenizer:
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# If we want to have a quick look at how it preprocesses the inputs, we can call the pre_tokenize_str method:
print(tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))

# We used the same default as for GPT-2 for the prefix space, so you can see that each word gets an initial 'Ġ' added at the beginning, except the first one.
# We can now train our tokenizer! This time we use a BpeTrainer.
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# To finish the whole pipeline, we have to include the post-processor and decoder:
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

# And like before, we finish by wrapping this in a Transformers tokenizer object:
from transformers import GPT2TokenizerFast

new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)