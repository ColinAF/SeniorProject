### NOTES ###
# - Add a module that keeps track of hyperparams and paths in json
# - Add more data transforms!!! 
# - Files for {train,validate,model,visualizations}
# - Super helpful tutorial: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
### NOTES ###

### Produce Detector ###

### External Imports ###
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
### External Imports ###

### All this is still copied from the tutorial for testing! - Not Mine ###

def get_model(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
### All this is still copied from the tutorial for testing! - Not Mine ###

### Produce Detector ###