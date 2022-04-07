### Produce Detector ###

### External Imports ###
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
### External Imports ###

### All this is still copied from the tutorial for testing! - Not Mine ###

# Probably better to make this an object and not a function?
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
### All this is still copied from the tutorial for testing! - Not Mine ###

### Produce Detector ###