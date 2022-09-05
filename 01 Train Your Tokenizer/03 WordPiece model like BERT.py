from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

# Let's have a look at how we can create a WordPiece tokenizer like the one used for training BERT. The first step is to create a Tokenizer with an empty WordPiece model:
# This tokenizer is not ready for training yet. We have to add some preprocessing steps: 
# the normalization (which is optional) and the pre-tokenizer, which will split inputs into the chunks we will call words. 
# The tokens will then be part of those words (but can't be larger than that).
tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))
# In the case of BERT, the normalization is lowercasing. Since BERT is such a popular model, it has its own normalizer: 
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# If you want to customize it, you can use the existing blocks and compose them in a sequence: here for instance we lower case, apply NFD normalization and strip the accents:
# https://huggingface.co/docs/tokenizers/components

# NFD Normalization : 코드를 정준 분해, 
# 01 발음 구별 기호가 붙은 글자가 하나로 처리되어 있을 경우, 이를 기호별로 나누어 처리,
# À (U+00C0) → A (U+0041) + ◌̀ (U+0300) 
# 02 한글을 한글 소리마디 영역(U+AC00~U+D7A3)으로 썼을 경우, 이를 첫가끝 코드로 처리
# 위 (U+C704) → ᄋ (U+110B) + ᅱ (U+1171) 
# 03 표준과 다른 조합 순서를 제대로 맞추기
# e (U+0071) + ◌̇ (U+0307) + ◌̣ (U+0323) → e (U+0071) + ◌̣ (U+0323) + ◌̇ (U+0307)

# Lowercase : 소문자화

# StripAccents : Removes all accent symbols in unicode (to be used with NFD for consistency)
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
# Like for the normalizer, we can combine several pre-tokenizers in a Sequence.
# If we want to have a quick look at how it preprocesses the inputs, we can call the pre_tokenize_str method:
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# There is also a BertPreTokenizer we can use directly. It pre-tokenizes using white space and punctuation:
print(tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))

# Note that the pre-tokenizer not only split the text into words but keeps the offsets, that is the beginning and start of each of those words inside the original text. 
# This is what will allow the final tokenizer to be able to match each token to the part of the text that it comes from 
# (a feature we use for question answering or token classification tasks).
# We can now train our tokenizer (the pipeline is not entirely finished but we will need a trained tokenizer to build the post-processor), 
# we use a WordPieceTrainer for that. The key thing to remember is to pass along the special tokens to the trainer, as they won't be seen in the corpus.
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# To actually train the tokenizer, the method looks like what we used before: we can either pass some text files, or an iterator of batches of texts:
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# Now that the tokenizer is trained, we can define the post-processor: 
# we need to add the CLS token at the beginning and the SEP token at the end (for single sentences) or several SEP tokens (for pairs of sentences).
# We use a TemplateProcessing to do this, which requires to know the IDs of the CLS and SEP token (which is why we waited for the training).
# So let's first grab the ids of the two special tokens:
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

# And here is how we can build our post processor. 
# We have to indicate in the template how to organize the special tokens with one sentence ($A) or two sentences ($A and $B).
# The : followed by a number indicates the token type ID to give to each part.

# Provides a way to specify templates in order to add the special tokens to each input sequence as relevant.
# Let's take BERT tokenizer as an example. It uses two special tokens, used to delimitate each sequence. 
# [CLS] is always used at the beginning of the first sequence, and [SEP] is added at the end of both the first, and the pair sequences. 
# The final result looks like this:
# Single sequence: [CLS] Hello there [SEP]
# Pair sequences: [CLS] My name is Anthony [SEP] What is my name? [SEP]
# With the type ids as following:
# [CLS]   ...   [SEP]   ...   [SEP]
#   0      0      0      1      1

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)

# We can check we get the expected results by encoding a pair of sentences for instance:
encoding = tokenizer.encode("This is one sentence.", "With this one we have a pair.")
# We can look at the tokens to check the special tokens have been inserted in the right places:
print(encoding.tokens)

# And we can check the token type ids are correct:
print(encoding.type_ids)

# The last piece in this tokenizer is the decoder, we use a WordPiece decoder and indicate the special prefix ##:
tokenizer.decoder = decoders.WordPiece(prefix="##")

# Now that our tokenizer is finished, we need to wrap it inside a Transformers object to be able to use it with the Transformers library.
# More specifically, we have to put it inside the class of tokenizer fast corresponding to the model we want to use, here a BertTokenizerFast:
from transformers import BertTokenizerFast

new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
# And like before, we can use this tokenizer as a normal Transformers tokenizer, and use the save_pretrained or push_to_hub methods.
# If the tokenizer you are building does not match any class in Transformers because it's really special, you can wrap it in PreTrainedTokenizerFast.