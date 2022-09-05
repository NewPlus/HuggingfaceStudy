from datasets import load_dataset

# For this example, we will use Wikitext-2 (which contains 4.5MB of texts so training goes fast for our example)
# but you can use any dataset you want (and in any language, just not English).
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

# We can have a look at the dataset, which as 36,718 texts:
print(dataset)
# To access an element, we just have to provide its index:
print(dataset[1])
# We can also access a slice directly, in which case we get a dictionary with the key "text" and a list of texts as value:
print(dataset[:5])

# The API to train our tokenizer will require an iterator of batch of texts, for instance a list of list of texts:
batch_size = 1000
all_texts = [dataset[i : i + batch_size]["text"] for i in range(0, len(dataset), batch_size)]
print(all_texts)

# To avoid loading everything into memory (since the Datasets library keeps the element on disk and only load them in memory when requested),
# we define a Python iterator. This is particularly useful if you have a huge dataset:
def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]

for text in batch_iterator():
    print(text)