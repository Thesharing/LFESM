import json
from typing import Tuple

import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from models.feature import extract_features_and_replace


class TripletTextDataset(Dataset):
    def __init__(self, text_a_list, text_b_list, text_c_list, label_list=None):
        if label_list is None or len(label_list) == 0:
            label_list = [None] * len(text_a_list)
        assert all(
            len(label_list) == len(text_list)
            for text_list in [text_a_list, text_b_list, text_c_list]
        )
        self.text_a_list = text_a_list
        self.text_b_list = text_b_list
        self.text_c_list = text_c_list
        self.label_list = [0 if label == "B" else 1 for label in label_list]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        text_a, text_b, text_c, label = (
            self.text_a_list[index],
            self.text_b_list[index],
            self.text_c_list[index],
            self.label_list[index],
        )
        return text_a, text_b, text_c, label

    @classmethod
    def from_dataframe(cls, df):
        text_a_list = df["A"].tolist()
        text_b_list = df["B"].tolist()
        text_c_list = df["C"].tolist()
        if "label" not in df:
            df["label"] = "B"
        label_list = df["label"].tolist()
        return cls(text_a_list, text_b_list, text_c_list, label_list)

    @classmethod
    def from_dict_list(cls, data, use_augment=False):
        df = pd.DataFrame(data)
        if "label" not in df:
            df["label"] = "B"
        if use_augment:
            df = TripletTextDataset.augment(df)
        return cls.from_dataframe(df)

    @classmethod
    def from_jsons(cls, json_lines_file, use_augment=False):
        with open(json_lines_file, 'r', encoding="utf-8") as f:
            data = list(map(lambda line: json.loads(line), f))
        return cls.from_dict_list(data, use_augment)

    @staticmethod
    def augment(df):
        df_cp1 = df.copy()
        df_cp1["B"] = df["C"]
        df_cp1["C"] = df["B"]
        df_cp1["label"] = "C"

        df = pd.concat([df, df_cp1])

        df = df.drop_duplicates()
        df = df.sample(frac=1)

        return df


def get_collator(max_len, device, tokenizer):
    def three_pair_collate_fn(batch):
        """
        Get a mini batch, convert the triplet into tensor.
        """
        example_tensors = []
        for text_a, text_b, text_c, label in batch:
            input_example = InputExample(text_a, text_b, text_c, label)
            a_feature, b_feature, c_feature = input_example.to_two_pair_feature(tokenizer, max_len)
            a_tensor, b_tensor, c_tensor = (
                a_feature.to_tensor(device),
                b_feature.to_tensor(device),
                c_feature.to_tensor(device)
            )
            label_tensor = torch.LongTensor([label]).to(device)
            example_tensors.append((a_tensor, b_tensor, c_tensor, label_tensor))

        return default_collate(example_tensors)

    return three_pair_collate_fn


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, features):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.features = features

    def to_tensor(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.LongTensor(self.input_ids).to(device),
            torch.LongTensor(self.segment_ids).to(device),
            torch.LongTensor(self.input_mask).to(device),
            torch.FloatTensor(self.features).to(device),
            torch.LongTensor([1]).to(device)
        )


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, text_a, text_b=None, text_c=None, label=None):
        """
        Constructs a InputExample.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

    @staticmethod
    def _text_pair_to_feature(text, tokenizer, max_seq_length):
        text, features = extract_features_and_replace(text)
        tokens = tokenizer.tokenize(text)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[len(tokens) - (max_seq_length - 2):]

        # https://huggingface.co/transformers/model_doc/bert.html?highlight=bertmodel#transformers.BertModel
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, segment_ids, input_mask, features

    def to_two_pair_feature(self, tokenizer, max_seq_length) -> Tuple[InputFeatures, InputFeatures, InputFeatures]:
        a = self._text_pair_to_feature(self.text_a, tokenizer, max_seq_length)
        b = self._text_pair_to_feature(self.text_b, tokenizer, max_seq_length)
        c = self._text_pair_to_feature(self.text_c, tokenizer, max_seq_length)
        return InputFeatures(*a), InputFeatures(*b), InputFeatures(*c)
