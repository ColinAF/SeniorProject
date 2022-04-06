import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Custom produce Dataset object 
class ProduceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        
        assert(os.path.exists(root)), "Path: " + root + " does not exist!"
        self.image_files = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(image_path) #.convert("RGB") # Don't forget to convert to RGB 
        image = transforms.ToTensor()(image)

        boxes = []

        labels = 0

        image_id = 0

        area = 0 

        iscrowd = 0 

        # A dictionary with the required feilds for pycocotools 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target 

    def __len__(self):
        return(len(self.image_files))