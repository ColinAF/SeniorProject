### NOTES ###
# - Add a module that keeps track of hyperparams and paths in json
# - Add more data transforms!!! 
# - Files for {train,validate,model,visualizations}
# - Super helpful tutorial: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
### NOTES ###

### Produce Detector ###

### External Imports ###
import torch # Get more specific things 
import torch.utils
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
### External Imports ###

### Local Imports ### 
from data_utils import DATASETS_PATH # Put pathnames and hyperparams in JSON 
from produce_dataset import ProduceDataset  
### Local Imports ### 

# Put each step in its own file (module)! 

# Load custom dataset 
train_dataset = ProduceDataset(root=DATASETS_PATH, 
                              annotations="assets/datasets/fruit_test/annotations.json", 
                              transforms=transforms.ToTensor())


### All this is still copied from the tutorial for testing! - Not Mine ###
# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

train_batch_size = 1

#Put the data in a dataloader
train_dataloader = DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              shuffle=True,
                              num_workers=4,
                              collate_fn= collate_fn) # Put all this in hyperparams

#train_features, train_labels = next(iter(train_dataloader))
#print(train_features)


# select device (whether GPU or CPU)
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu') # Training uses way too much GPU atm see if I can manage this better!

# # DataLoader is iterable over Dataset
# for imgs, annotations in train_dataloader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
    
## Add time info and epoch count! also this should go in a training script ##
# 2 classes; Only target class or background
num_classes = 3
num_epochs = 10
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)
    
# parameters
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

len_dataloader = len(train_dataloader)

for epoch in range(num_epochs):
    model.train()
    i = 0    
    for imgs, annotations in train_dataloader :
        i += 1
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
## Add time info and epoch count! also this should go in a training script ##

### All this is still copied from the tutorial for testing! - Not Mine ###


# Define the model

# Train model 

# Test model

### Produce Detector ###