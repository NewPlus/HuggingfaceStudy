from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

# Let's now have a look at how we can create a Unigram tokenizer like the one used for training T5. The first step is to create a Tokenizer with an empty Unigram model:
tokenizer = Tokenizer(models.Unigram())

# Like before, we have to add the optional normalization (here some replaces and lower-casing) and we need to specify a pre-tokenizer before training. 
# The pre-tokenizer used is a Metaspace pre-tokenizer: 
# it replaces all spaces by a special character (defaulting to ▁) and then splits on that character.
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.Replace("``", '"'), normalizers.Replace("''", '"'), normalizers.Lowercase()]
)
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# If we want to have a quick look at how it preprocesses the inputs, we can call the pre_tokenize_str method:
print(tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))

# You can see that each word gets an initial ▁ added at the beginning, as is usually done by sentencepiece.
# We can now train our tokenizer! This time we use a UnigramTrainer.
# "We have to explicitely set the unknown token in this trainer otherwise it will forget it afterward.
trainer = trainers.UnigramTrainer(vocab_size=25000, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# To finish the whole pipeline, we have to include the post-processor and decoder. 
# The post-processor is very similar to what we saw with BERT, the decoder is just Metaspace, like for the pre-tokenizer.
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)
tokenizer.decoder = decoders.Metaspace()

# And like before, we finish by wrapping this in a Transformers tokenizer object:
from transformers import AlbertTokenizerFast

new_tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)