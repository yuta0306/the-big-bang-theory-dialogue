import argparse
import json
import re
from random import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_json(path: str):
    data = json.load(open(path, "r"))
    return data


def preprocess(data: list):
    new_data = []
    for episode in data:
        title = episode["title"]
        scripts: list = episode["scripts"]
        scripts = [item for item in scripts if len(item) == 2]
        scripts = [
            [
                re.sub(r"\(.*\)", "", character).strip(),
                re.sub(r"\(.*\)", "", utterance).strip(),
            ]
            for (character, utterance) in scripts
        ]
        new_data.append(
            {
                "title": title,
                "scripts": scripts,
            }
        )
    return new_data


def split_data(data: list, ratio: float = 0.7):
    episodes = len(data)
    train_data = data[: int(episodes * ratio)]
    test_data = data[int(episodes * ratio) :]

    return train_data, test_data


class TheBigBangTheoryDataset(Dataset):
    def __init__(self, data: list):
        super(TheBigBangTheoryDataset, self).__init__()
        self._build(data)

    def _build(self, data: list):
        print("build dataset...")
        corpus = []
        for episode in tqdm(data):
            # title = episode["title"]
            scripts = episode["scripts"]
            for idx in range(2, len(scripts) - 1):
                text = ""
                for uttr in scripts[max(idx - 7, 0) : idx]:
                    text += ": ".join(uttr)
                target = scripts[idx][0] + ":"  # this is a chatee
                gold = scripts[idx][0] + ": " + scripts[idx][1]
                corpus.append([text, gold, target])
        self.data = corpus

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        inputs, gold, target = self.data[index]
        return inputs, gold, target


class CollateFn:
    def __init__(self, tokenizer, max_length: int = 128):
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        inputs, gold, target = zip(*batch)
        inputs, gold, target = list(inputs), list(gold), list(target)
        inputs = self.tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        gold = self.tokenizer(
            gold,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        target = self.tokenizer(
            target,
            padding="max_length",
            truncation=True,
            max_length=5,
            return_tensors="pt",
        )
        return inputs, gold, target


if __name__ == "__main__":
    # prepare dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./best.ckpt")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument(
        "--tokenizer_path", type=str, default="the-big-bang-theory-dialogue"
    )
    parser.add_argument("--add_tokens", type=bool, default=False)
    args = parser.parse_args()

    data = load_json("data.json")
    data = preprocess(data=data)
    train_data, test_data = split_data(data=data)

    if args.add_tokens:
        new_vocab = []
        for episode in data:
            for vocab, _ in episode["scripts"]:
                new_vocab.append(vocab)
        # counter = Counter(new_vocab)
        # vocab = set()
        # for key, value in counter.items():
        #     if value > 10:
        #         vocab.add(key)
        vocab = list(set(new_vocab))

        print("New vocab:", vocab)

    train_dataset = TheBigBangTheoryDataset(data=train_data)
    # print(train_dataset[0])
    test_dataset = TheBigBangTheoryDataset(data=test_data)
    # print(test_dataset[0])

    # Train
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.add_tokens:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        print("Before Resize Vocab:", len(tokenizer))
        print("Size Vocab:", len(vocab))
        tokenizer.add_tokens(list(vocab))
        print("After Resize Vocab:", len(tokenizer))
        print("=" * 30)
        tokenizer.save_pretrained(args.tokenizer_path)
        print("Save tokenizer!")
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
        print("Vocab size:", len(tokenizer))

    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=CollateFn(tokenizer=tokenizer),
    )
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        max_length=64,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-7)
    epoch = 15
    patience = 3
    best_loss = 1e10
    for e in range(epoch):
        print(f"=== EPOCH : {e} ===")
        loss_total = 0.0
        for i, (inputs, gold, target) in enumerate(tqdm(dataloader), 1):
            outputs = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                # decoder_input_ids=target.input_ids.to(device),
                # decoder_attention_mask=target.attention_mask.to(device),
                labels=gold.input_ids.to(device),
            )
            loss = outputs.loss
            loss_total += loss
            loss.backward()
            optimizer.step()
        loss_total /= i
        print(f"Loss on Epoch {e}:", loss_total)
        patience -= 1
        if loss_total < best_loss:
            best_loss = loss_total
            torch.save(model.state_dict(), args.ckpt)
            print("!!! UPDATE CHECKPOINT !!!")
            patience = 3
        if patience <= 0:
            break

    # Inference
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CollateFn(tokenizer=tokenizer),
    )
    with torch.no_grad():
        for inputs, gold, target in dataloader:
            outputs = model.generate(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                decoder_input_ids=target.input_ids.to(device),
                decoder_attention_mask=target.attention_mask.to(device),
                do_sample=False,
            )
            if random() > 0.8:
                print("=" * 20)
                print(
                    "INPUT:",
                    tokenizer.decode(
                        inputs["input_ids"][0],
                        skip_special_tokens=True,
                        do_sample=False,
                    ),
                )
                print(
                    "PREDICT:",
                    tokenizer.decode(
                        outputs.detach()[0],
                        skip_special_tokens=True,
                        do_sample=False,
                    ),
                )
                print(
                    "GOLD:",
                    tokenizer.decode(
                        gold["input_ids"][0],
                        skip_special_tokens=True,
                        do_sample=False,
                    ),
                )
                print("=" * 20)
