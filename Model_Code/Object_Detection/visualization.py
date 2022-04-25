### Visualizations ###

### External Imports ###
import torch
import csv
import numpy as np
import torchvision
import matplotlib.pyplot as plt 
from torchvision.utils import draw_bounding_boxes
### External Imports ###

# Evaluate an image! 
def visualize_predictions(dataloader, model=None):
    # Display images and label.
    device = torch.device('cpu')
    score_threshold = .10     

    inst_classes = [
        '_background_', 'apple', 'kiwi', 'lemon', 'banana', 'lime', 'tangerine', 'garlic', 'avocado'
    ]

    imgs = []

    if model is not None: 
        model.to(device)
        model.eval()

    for images, annotations in dataloader:
        if model is not None: 
            images = list(image.to(device) for image in images)
            outputs = model(images)

            lb = {}

            for i in range(len(outputs[0]['labels'])):
                lb[i] = inst_classes[outputs[0]['labels'][i]]

            img = torchvision.transforms.ConvertImageDtype(torch.uint8)(images[0])
            img = draw_bounding_boxes(img, boxes=outputs[0]['boxes'][outputs[0]['scores'] > score_threshold], labels=lb, colors="green", width=4)
        else: 
            lb = {}
            
            for i in range(len(annotations[0]['labels'])):
                lb[i] = inst_classes[annotations[0]['labels'][i]]
                
            img = torchvision.transforms.ConvertImageDtype(torch.uint8)(images[0])
            img = draw_bounding_boxes(img, boxes=annotations[0]['boxes'], labels=lb, colors="green", width=4)
            
        imgs.append(img)

    imshow(torchvision.utils.make_grid(imgs[0:4]))


def displayImage():
    pass 

def imshow(img):
    img = img      # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Would be cool to have a function that plotted losses
def plotLoses(stats_file):
    stats = open(stats_file, 'r')
    read_stats = csv.DictReader(stats)

    losses = np.empty(0)
    step = np.empty(0)

    i = 1
    for lines in read_stats:
        losses = np.append(losses, float(lines['Loss']))
        step = np.append(step, (int(lines['Epoch']) * i))
        i += 1

    y = np.arange(start=1, stop=(len(losses)+1), step=1)

    plt.title("Loss Plot")
    plt.plot(y, losses, color="red")
    plt.show()

plotLoses('train_stats00.csv')

### Visualizations ###