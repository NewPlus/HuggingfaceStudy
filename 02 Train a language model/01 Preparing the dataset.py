import transformers
from huggingface_hub import notebook_login

notebook_login()
print(transformers.__version__)

# For each of those tasks, we will use the Wikitext 2 dataset as an example. You can load it very easily with the 🤗 Datasets library.
from datasets import load_dataset
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

# To access an actual element, you need to select a split first, then give an index
# print(datasets["train"][10])

# To get a sense of what the data looks like, the following function will show some examples picked randomly in the dataset.
from datasets import ClassLabel
import random
import pandas as pd

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    
    return df

print(show_random_elements(datasets["train"]))
# As we can see, some of the texts are a full paragraph of a Wikipedia article while others are just titles or empty lines.
