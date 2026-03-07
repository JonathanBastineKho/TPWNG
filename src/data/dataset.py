import torch
from torch.utils.data import Dataset
from pathlib import Path


class UCFCrimeDataset(Dataset):
    """
    Args:
        root:   Path to data/ucfcrime/train or data/ucfcrime/test
        normal: If True, load only Normal videos.
                If False, load only anomaly videos.
                If None, load everything.
    """

    def __init__(self, root: str, normal: bool = None):
        self.root    = Path(root)
        self.samples = []   # list of (pt_path, class_name)

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue

            cls       = class_dir.name
            is_normal = cls.lower() == 'normal'

            if normal is True  and not is_normal:
                continue
            if normal is False and is_normal:
                continue

            for pt_file in sorted(class_dir.glob('*.pt')):
                self.samples.append((pt_file, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pt_path, cls = self.samples[index]
        feat = torch.load(pt_path, map_location='cpu', weights_only=True)
        return feat, cls