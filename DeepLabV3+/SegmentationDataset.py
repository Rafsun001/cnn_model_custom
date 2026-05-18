from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.image_paths = self.collect_image_paths()
        self.mask_paths = self.match_masks_to_images()

    def collect_image_paths(self):
        allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        image_paths = []

        for path in sorted(self.images_dir.iterdir()):
            if path.suffix.lower() in allowed_extensions:
                image_paths.append(path)

        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in: {self.images_dir}")

        return image_paths

    def match_masks_to_images(self):
        allowed_mask_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

        mask_paths = []

        for image_path in self.image_paths:
            matched_mask_path = None

            for extension in allowed_mask_extensions:
                possible_mask_path = self.masks_dir / f"{image_path.stem}{extension}"

                if possible_mask_path.exists():
                    matched_mask_path = possible_mask_path
                    break

            if matched_mask_path is None:
                raise RuntimeError(
                    f"No matching mask found for image: {image_path.name}"
                )

            mask_paths.append(matched_mask_path)

        return mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")


        mask = Image.open(mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
