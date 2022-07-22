import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk import sent_tokenize
import sentencepiece as spm
from config import *


CLS = 3
SEP = 4
MASK = 5  # 18, 5
PAD = 6


def flatten(l):
    return [x for s in l for x in s]


class SPMDataset(Dataset):
    def __init__(self, name, articles, max_len=MAX_LEN):
        self.name = name
        self.max_len = max_len
        list_of_list_of_sentences = list(
            map(
                lambda x: [x["title"]] + sent_tokenize(x["text"].replace("\n", ".")),
                articles,
            )
        )
        self.sentences = list(
            filter(
                lambda x: x.count(" ") < (self.max_len // 3),
                flatten(list_of_list_of_sentences),
            )
        )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class WikiDataset(Dataset):
    def __init__(self, name, articles, sp, max_len=MAX_LEN):
        self.name = name
        self.sp = sp
        self.max_len = max_len
        list_of_list_of_sentences = list(
            map(
                lambda x: [x["title"]] + sent_tokenize(x["text"].replace("\n", ".")),
                articles,
            )
        )
        self.sentences = list(
            filter(
                lambda x: x.count(" ") < (self.max_len // 3),
                flatten(list_of_list_of_sentences),
            )
        )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        if random.random() < 0.5:
            next_sentence = self.sentences[(idx + 1) % len(self)]
            is_next_label = 1
        else:
            next_sentence = self.sentences[random.randint(0, len(self))]
            is_next_label = 0

        encoding = self.sp.Encode(sentence, out_type=int)
        next_encoding = self.sp.Encode(next_sentence, out_type=int)

        bert_input = (
            [CLS]
            + [get_random_choice(x) for x in encoding]
            + [SEP]
            + [get_random_choice(x) for x in next_encoding]
            + [SEP]
        )

        bert_input = bert_input + [PAD] * (self.max_len - len(bert_input))
        bert_label = [CLS] + encoding + [SEP] + next_encoding + [SEP]
        bert_label = bert_label + [PAD] * (self.max_len - len(bert_label))

        segment_label = [1 for _ in range(len(encoding) + 2)] + [
            2 for _ in range(len(next_encoding) + 1)
        ]
        segment_label = segment_label + [0] * (self.max_len - len(segment_label))

        output = {
            "bert_input": ttt(bert_input, self.max_len),
            "bert_label": ttt(bert_label, self.max_len),
            "segment_label": ttt(segment_label, self.max_len),
            "is_next": is_next_label,
        }

        return output


def train_spm():
    filename = "data/data.m3.txt"

    dataset = load_dataset("wikipedia", "20220301.simple")
    train, test_val = train_test_split(list(dataset["train"]), test_size=0.6)
    val, test = train_test_split(test_val, test_size=0.5)
    train_dataset = SPMDataset("train", train)
    val_dataset = SPMDataset("val", val)
    test_dataset = SPMDataset("test", test)

    with open(filename, "w", encoding="UTF-8") as w:
        for loader in [train_dataset, val_dataset, test_dataset]:
            for item in loader:
                w.write(item)
                w.write("\n")

    # Save data to file
    spm.SentencePieceTrainer.train(
        input=filename,
        model_prefix="data/m3",
        vocab_size=VOCAB_SIZE,
        user_defined_symbols=["<cls>", "<sep>", "<mask>", "<pad>"],
    )


def load_dataloaders():
    sp = spm.SentencePieceProcessor(model_file="data/m3.model")
    dataset = load_dataset("wikipedia", "20220301.simple")
    train, test_val = train_test_split(list(dataset["train"]), test_size=0.6)
    val, test = train_test_split(test_val, test_size=0.5)

    train_dataset = WikiDataset("train", train, sp)
    val_dataset = WikiDataset("val", val, sp)
    test_dataset = WikiDataset("test", test, sp)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return (train_dataloader, val_dataloader, test_dataloader)


def get_random_choice(word: int) -> int:
    if random.random() < 0.85:
        return word

    choice = random.random()
    if choice < 0.1:
        return word

    if choice < 0.2:
        return random.randint(7, VOCAB_SIZE)

    return MASK


def ttt(l, s):
    return torch.tensor(l[:s])


if __name__ == "__main__":
    import torch

    nltk.download("punkt")
    train_spm()

    train_dataloader, _, _ = load_dataloaders()
    xx = next(iter(train_dataloader))
