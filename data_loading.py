import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMG_DIR, TEXT_DIR, MAX_LEN

class ResizePadTransform:
    def __init__(self, target_size):
        self.target_size = target_size
        
    def __call__(self, image):
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        resize_transform = transforms.Resize((new_height, new_width))
        resized_image = resize_transform(image)

        pad_width = self.target_size - new_width
        pad_height = self.target_size - new_height
        
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        padding_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom))
        padded_resized_image = padding_transform(resized_image)

        return padded_resized_image

# Define transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
tfms = transforms.Compose([
    ResizePadTransform(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, data_split, transform=None, tokenizer=None, max_len=MAX_LEN, phase="train"):
        self.img_folder = Path(root_dir, "images")
        self.text_folder = Path(root_dir, "text")
        split_file = Path(self.text_folder, f"Flickr_8k.{data_split}Images.txt")

        self.images = pd.read_csv(split_file, names=["image_name"])
        self.captions = pd.read_csv(
            Path(self.text_folder, "Flickr8k.token.txt"),
            names=["image_name", "caption"],
            sep="\t"
        )

        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.phase = phase

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.phase == "train":
            caption_row = self.captions.iloc[idx]
            caption = caption_row["caption"]
            caption = f"{self.tokenizer.bos_token} {caption} {self.tokenizer.eos_token}"
            caption = self.tokenizer.encode(caption)

            pad_tokens = self.max_len - len(caption) + 1
            caption += pad_tokens * [self.tokenizer.pad_token_id]
            caption = torch.tensor(caption)

            image_name = caption_row["image_name"][:-2]
            image = Image.open(Path(self.img_folder, image_name))
            image = self.transform(image)
        else:
            image_name = self.images["image_name"][idx]
            image = Image.open(Path(self.img_folder, image_name))
            image = self.transform(image)

            caption = self.captions[self.captions["image_name"].str.contains(
                image_name)]["caption"].tolist()

        return {"image": image, "caption": caption, "image_name": image_name}