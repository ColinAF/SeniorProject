
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

fruit2_int = read_image(str(Path('assets') / 'Fruit02.jpg'))


from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype

# For monitoring stuff 
import time 
import os 

# As of now most of the important stuff happens here 
os.system("nvidia-smi --query-gpu=memory.used --format=csv")

try: 
    t0 = time.time()

    # Inputs and model need to be on the GPU
    fruit2_int = fruit2_int.to(device)

    batch_int = torch.stack([fruit2_int])

    batch = convert_image_dtype(batch_int, dtype=torch.float)

    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)

    model = model.eval() 

    # Inputs and model need to be on the GPU
    model = model.to(device)

    outputs = model(batch)

    t1 = time.time()

    print(t1 - t0)
    
except RuntimeError:
    print("Out of GPU Memory")
    torch.cuda.empty_cache() # Seems like this is useless 
    os.system("nvidia-smi --query-gpu=memory.used --format=csv")

#print(len(outputs[0]['labels']))

os.system("nvidia-smi --query-gpu=memory.used --format=csv")

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

lb = {}

outputs[0]['boxes'] = outputs[0]['boxes'].to("cpu")

for i in range(len(outputs[0]['labels'])):
    lb[i] = inst_classes[outputs[0]['labels'][i]]

score_threshold = .7
fruit_with_boxes = [
    draw_bounding_boxes(apple_int, boxes=output['boxes'][output['scores'] > score_threshold], labels=lb, width=4)
    for apple_int, output in zip(batch_int, outputs)
]

show(fruit_with_boxes)
