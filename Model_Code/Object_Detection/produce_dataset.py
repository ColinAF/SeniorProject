### Produce Dataset ###

### External Imports ###
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
### External Imports ###


# Custom produce Dataset object 
class ProduceDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, transforms):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))
        

    def __getitem__(self, index):
        coco = self.coco

        image_id = self.ids[index]

        image_path = coco.loadImgs(image_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, image_path))

        # Annotations hold bounding box info 
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotation = coco.loadAnns(annotation_ids)

        objects_in_image = len(annotation)

        # Bounding box coordinates
        boxes = []

        # COCO -> [xmin, ymin, width, height]
        # PyTorch -> [xmin, ymin, xmax, ymax]
        for i in range(objects_in_image):
            xmin = annotation[i]['bbox'][0]
            ymin = annotation[i]['bbox'][1]
            xmax = xmin + annotation[i]['bbox'][2]
            ymax = ymin + annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)     

        labels = torch.ones((objects_in_image,), dtype=torch.int64)

        image_id = torch.tensor([image_id])

        areas = []

        for i in range(objects_in_image):
                areas.append(annotation[i]['area'])

        areas = torch.as_tensor(areas, dtype=torch.float64)
    
        iscrowd = torch.zeros((objects_in_image,), dtype=torch.int64)

        # A dictionary with the required feilds for pycocotools 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target 

    def __len__(self):
        return(len(self.ids))

### Produce Dataset ###