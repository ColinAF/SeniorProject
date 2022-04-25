# Make this a class DUH!!! 
import torch
import torchvision

from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.io import read_image
from pathlib import Path

# COCO Classes! 
# inst_classes = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

inst_classes = [
    '_background_', 'apple', 'kiwi', 'lemon', 'banana', 'lime', 'tangerine', 'garlic', 'avocado'
]

class ObjectDetector: 
    # Load model into memory and choose a device
    def __init__(self):
        self.device = torch.device("cpu") # Make this platform agnostic 
        # Put the final version of the trained model here! 
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 9) # don't hard code this! ;)
        self.model.load_state_dict(torch.load('/home/colin/SeniorProject/detector.pt'))
        self.model.eval() 
        self.model.to(self.device)
        
    # Evaluate an image! 
    def run_model(self, name):
        test_int = read_image(str(Path('media/images') / name))
        test_int = test_int.to(self.device)

        # is it really worth making a batch? 
        batch_int = torch.stack([test_int]) 
        batch = convert_image_dtype(batch_int, dtype=torch.float)

        outputs = self.model(batch)
        lb = {}

        for i in range(len(outputs[0]['labels'])):
            lb[i] = inst_classes[outputs[0]['labels'][i]]

        score_threshold = .20

        im = draw_bounding_boxes(test_int, boxes=outputs[0]['boxes'][outputs[0]['scores'] > score_threshold], labels=lb, width=4) # windows doesn't like the color argument for some reason?
        im = torchvision.transforms.ToPILImage()(im)
        im.save("media/images/produce_bowl.jpg", "JPEG")

        produce = {}
        for i in lb:
            if outputs[0]['scores'][i] > score_threshold:
                
                class_value = produce.get(lb[i])

                if class_value == None:
                    produce.update({lb[i]: 1})  
                else:
                    class_value += 1
                    produce.update({lb[i]: class_value})

        return produce

