# coding=utf-8
"""NER Dataset for GlobalPointer.

Only ``NERDatasetForGP`` is kept — it reads JSONL samples produced by
``data_utils/process_ner_data.py`` and builds the GlobalPointer label matrix
in ``collate_wrapper``.
"""
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class NERDatasetForGP(Dataset):
    def __init__(self, file, ent2id_path):
        super(NERDatasetForGP, self).__init__()
        self.ent2id = json.load(open(ent2id_path, 'r', encoding='utf-8'))
        self.data = self._load_data(file)
        self.max_len = 512

    def _load_data(self, file: str):
        """
        Args:
            file (str): file path

        Returns:
            list: [{"input_ids": [...],
                    "labels": [...],
                    "token": [...],
                    "labels_": [...],
                    "entity": [[text, start, end_exclusive, type], ...]}, ...]
        """
        data = []
        errors = 0
        with open(file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                try:
                    if "entity" not in sample:
                        errors += 1
                        continue
                    data.append(sample)
                except ValueError:
                    errors += 1
                    continue
        random.shuffle(data)
        print("Num errors: ", errors)
        print("Total number of data: ", idx + 1)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item

    def encoder(self, item):
        raw_text = ''.join(item["token"])
        input_ids = item["input_ids"]
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        return (
            raw_text,
            input_ids,
            token_type_ids,
            attention_mask,
        )

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode="post"):
        """Numpy function, pad sequences to the same length."""
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, "__getitem__"):
            length = [length]

        slices = [np.s_[: length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == "post":
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == "pre":
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, "constant", constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate_wrapper(self, examples):
        (
            raw_text_list,
            batch_input_ids,
            batch_attention_mask,
            batch_labels,
            batch_segment_ids,
        ) = ([], [], [], [], [])

        for item in examples:
            (
                raw_text,
                input_ids,
                token_type_ids,
                attention_mask,
            ) = self.encoder(item)

            entity = item['entity']
            labels = np.zeros((len(self.ent2id), self.max_len, self.max_len))
            for _, start, end, label in entity:
                labels[self.ent2id[label], start, end - 1] = 1

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, : len(input_ids), : len(input_ids)])

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(
            self.sequence_padding(batch_attention_mask)
        ).float()
        batch_labels = torch.tensor(
            self.sequence_padding(batch_labels, seq_dims=3)
        ).long()

        samples = {
            "input_ids": batch_inputids,
            "labels": batch_labels,
            "attention_mask": batch_attentionmask,
            "raw_text_list": raw_text_list,
            "token_type_ids": batch_segmentids,
        }

        return samples
