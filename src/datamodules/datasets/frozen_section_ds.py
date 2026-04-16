import os
import sys

import torch
from torch.utils.data import Dataset
from typing import Any, Dict, Sequence, Tuple, List
import glob
from tqdm import tqdm



class CtranspathFrozenSectionDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = rows
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        self._load_data_in_ram()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]

    def _load_data_in_ram(self) -> None:
        root = "/z3/home/JadeSprings/frozenSection/result/embedding"
        for item in tqdm(self.rows, total=len(self)):
            slide = item["slide"]
            label = item["label"]
            search = os.path.join(root, slide, "*/embedding.pth")
            embedding_files = sorted(glob.glob(search), key=lambda x: int(x.split('/')[-2].split('_')[0]))
            input = torch.cat([torch.load(e, weights_only=True) for e in embedding_files], dim=0)

            self.inputs.append(input)
            self.labels.append(torch.tensor(int(label), dtype=torch.long))


            # fake_input = torch.randn([10, 768], dtype=torch.float32)
            # fake_label = torch.tensor(int(item["label"]), dtype=torch.long)
            # self.inputs.append(fake_input)
            # self.labels.append(fake_label)

class Resnet50FrozenSectionDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = rows
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        self._load_data_in_ram()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]

    def _load_data_in_ram(self) -> None:
        root = "/z3/home/JadeSprings/frozenSection/result/emb_resnet50"
        for item in tqdm(self.rows, total=len(self)):
            slide = item["slide"]
            label = item["label"]
            search = os.path.join(root, slide, "*/embedding.pth")
            embedding_files = sorted(glob.glob(search), key=lambda x: int(x.split('/')[-2].split('_')[0]))
            input = torch.cat([torch.load(e, weights_only=True) for e in embedding_files], dim=0)

            self.inputs.append(input)
            self.labels.append(torch.tensor(int(label), dtype=torch.long))

if __name__ == "__main__":
    csv_file = "/z3/home/JadeSprings/deeplab/data/frozen_section/manifest.csv"

    import csv
    with open(csv_file, "r") as f:
        reader = list(csv.DictReader(f))


    def print_memory_usage():
        """打印当前进程的内存使用"""
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()

        print(f"RSS (物理内存): {mem.rss / 1024 / 1024:.2f} MB")
        print(f"VMS (虚拟内存): {mem.vms / 1024 / 1024:.2f} MB")


    # 使用
    print_memory_usage()
    ds = CtranspathFrozenSectionDataset(reader)
    print_memory_usage()  # 可以看到实际增加了多少


