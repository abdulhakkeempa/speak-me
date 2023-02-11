import os
import pickle
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from torch.utils.data import Dataset


class UnicodeData(Dataset):
    def __init__(self, data_dir, block_size=128):
        self.data = self.get_tokens(data_dir)
        self.encode, self.decode = self.get_encodings()

        self.vocab_size = self.decode.shape[0]
        self.block_size = block_size

        print(f"Vocab size: {self.vocab_size}")

    @staticmethod
    def get_file_list(data_dir) -> Generator[Path, None, None]:
        for cpath, folders, files in os.walk(data_dir):
            for file in files:
                yield Path(cpath, file)

    def get_tokens(self, data_dir) -> np.ndarray:
        if os.path.exists("tokens.npy"):
            return np.load("tokens.npy", mmap_mode="r", encoding="bytes")

        file_list = self.get_file_list(data_dir)
        tokens = []

        for file in file_list:
            tokens += list(file.read_text(encoding="utf-8"))

        tokens = np.array(tokens)
        np.save("tokens.npy", tokens)

        return tokens

    def get_encodings(self):
        if os.path.exists("encode.pkl") and os.path.exists("decode.npy"):
            return (pickle.load(open("encode.pkl", "rb")),
                    np.load("decode.npy", mmap_mode="r", encoding="bytes"))

        #regex to filter malayalam,english,symbols,emojis.
        pattern = r'(?!.*[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u03E2-\u03EF\u2C80-\u2CF3])^([\u0D00-\u0D7F]+|[A-Za-z]+|[\u00A9|\u00AE|[\u2000-\u3300]|\uD83C[\uDF00-\uDFFF]|\uD83D[\uDC00-\uDE4F]|\uD83E[\uDD10-\uDDFF])+|[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff\U00002702-\U000027b0]'
        tokens = np.unique(self.data)
        encode = {ch: i for i, ch in enumerate(tokens)}
        
        #tokens after regex filtering.
        decode = [char for char in tokens if re.match(pattern,char) ]
        decode = np.array(decode)

        pickle.dump(encode, open("encode.pkl", "wb"))
        np.save("decode.npy", decode)

        return encode, decode

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def encode_sequence(self, sequence):
        return [self.encode[s] for s in sequence]

    def decode_sequence(self, sequence):
        return "".join([self.decode[s] for s in sequence])

    def __len__(self):
        return self.data.shape[0] - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every token to an integer
        dix = [self.encode[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    UnicodeData("data", 1024)
