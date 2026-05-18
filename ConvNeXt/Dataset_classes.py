from torch.utils.data import Dataset
from datasets import load_dataset


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, transform=None):
        self.hf_dataset = load_dataset(
            "Maysee/tiny-imagenet",
            split="train",
        )
        self.transform = transform
        self.class_to_idx = self._get_class_to_idx()

    def _get_class_to_idx(self):
        labels = sorted(set(self.hf_dataset["label"]))
        return {str(label): label for label in labels}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        sample = self.hf_dataset[index]

        image = sample["image"]
        label = sample["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class TinyImageNetValDataset(Dataset):
    def __init__(self, transform=None):
        self.hf_dataset = load_dataset(
            "Maysee/tiny-imagenet",
            split="valid",
        )
        self.transform = transform
        self.class_to_idx = self._get_class_to_idx()

    def _get_class_to_idx(self):
        labels = sorted(set(self.hf_dataset["label"]))
        return {str(label): label for label in labels}

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        sample = self.hf_dataset[index]

        image = sample["image"]
        label = sample["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
