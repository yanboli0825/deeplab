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

class MetaCtranspathFrozenSectionDataset(Dataset):
    MAG_TO_ID = {
        2: 0,
        4: 1,
        10: 2,
        20: 3,
        40: 4,
    }

    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = rows
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.order: List[torch.Tensor] = []
        self.meta: List[torch.Tensor] = []
        self.mag: List[torch.Tensor] = []
        self.patch_mask: List[torch.Tensor] = []

        self._load_data_in_ram()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.order[idx],
            self.meta[idx],
            self.mag[idx],
            self.patch_mask[idx],
        )

    def _load_data_in_ram(self) -> None:
        root = "/z3/home/JadeSprings/frozenSection/result/embedding"
        for item in tqdm(self.rows, total=len(self)):
            slide = item["slide"]
            label = item["label"]
            search = os.path.join(root, slide, "*/embedding.pth")
            embedding_files = sorted(glob.glob(search), key=lambda x: int(x.split('/')[-2].split('_')[0]))
            if not embedding_files:
                raise ValueError(f"No embedding files found for slide '{slide}'")

            frame_tensors: List[torch.Tensor] = []
            order: List[int] = []
            mag: List[int] = []
            clear: List[float] = []
            area: List[float] = []
            for f in embedding_files:
                frame_tensor = torch.load(f, weights_only=True)
                if frame_tensor.ndim == 1:
                    frame_tensor = frame_tensor.unsqueeze(0)
                if frame_tensor.ndim != 2:
                    raise ValueError(
                        f"Frame embedding must be 2D [P, D], got shape={tuple(frame_tensor.shape)} for '{f}'"
                    )
                frame_tensors.append(frame_tensor)

                frame_dir = os.path.basename(os.path.dirname(f))
                frame_meta = frame_dir.split('_')
                order.append(int(frame_meta[0]))
                clear.append(float(frame_meta[1]))
                area.append(float(frame_meta[3]))
                raw_mag = int(frame_meta[4])
                if raw_mag not in self.MAG_TO_ID:
                    raise ValueError(
                        f"Unsupported magnification '{raw_mag}' for slide '{slide}' in frame '{frame_dir}'. "
                        f"Supported magnifications are {sorted(self.MAG_TO_ID)}."
                    )
                mag.append(self.MAG_TO_ID[raw_mag])

            num_frames = len(frame_tensors)
            max_patches = max(frame.shape[0] for frame in frame_tensors)
            dim = frame_tensors[0].shape[1]

            x_padded = torch.zeros(num_frames, max_patches, dim, dtype=frame_tensors[0].dtype)
            patch_mask = torch.ones(num_frames, max_patches, dtype=torch.bool)

            for frame_idx, frame_tensor in enumerate(frame_tensors):
                if frame_tensor.shape[1] != dim:
                    raise ValueError(
                        f"Inconsistent embedding dim for slide '{slide}': expected {dim}, got {frame_tensor.shape[1]}"
                    )
                patch_count = frame_tensor.shape[0]
                x_padded[frame_idx, :patch_count] = frame_tensor
                patch_mask[frame_idx, :patch_count] = False

            order_tensor = torch.tensor(order, dtype=torch.long)
            mag_tensor = torch.tensor(mag, dtype=torch.long)
            clear_tensor = torch.log1p(torch.tensor(clear, dtype=torch.float32))
            area_tensor = torch.log1p(torch.tensor(area, dtype=torch.float32))
            norm_order = order_tensor.float()
            norm_order = norm_order / max(int(order_tensor.max().item()), 1)
            meta_tensor = torch.stack([clear_tensor, area_tensor, norm_order], dim=-1)

            self.inputs.append(x_padded)
            self.labels.append(torch.tensor(int(label), dtype=torch.long))
            self.order.append(order_tensor)
            self.meta.append(meta_tensor)
            self.mag.append(mag_tensor)
            self.patch_mask.append(patch_mask)


if __name__ == "__main__":
    csv_file = "/z3/home/JadeSprings/deeplab/data/frozen_section/manifest.csv"

    import csv
    with open(csv_file, "r") as f:
        reader = list(csv.DictReader(f))

    import sys
    ds = MetaCtranspathFrozenSectionDataset(reader[:5])
    from src.models.cpath.ours import MOHPMIL_v0

    model_cfg = {"dim_in": 768, "dim_hidden": 1024, "num_mag": 5, "num_classes": 2, "topk_ratio": 0.2}
    model = MOHPMIL_v0(model_cfg)
    x, y, order, meta, mag_id, patch_mask = ds[0]
    with torch.no_grad():
        logits = model(
            x.unsqueeze(0),
            order.unsqueeze(0),
            meta.unsqueeze(0),
            mag_id.unsqueeze(0),
            patch_mask.unsqueeze(0),
        )
        print(logits.shape)
    sys.exit(0)


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

