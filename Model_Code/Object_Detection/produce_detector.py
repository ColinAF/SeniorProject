### NOTES ###
# - Add a module that keeps track of hyperparams and paths in json
# - Add more data transforms!!! 
# - Files for {train,validate,model,visualizations}
# - Super helpful tutorial: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
### NOTES ###

### Produce Detector ###

### External Imports ###
import torch # Get more specific things 
from torch.utils.data import DataLoader
from torchvision import transforms
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

# Put the data in a dataloader
train_dataloader = DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn) # Put all this in hyperparams


# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DataLoader is iterable over Dataset
for imgs, annotations in train_dataloader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)

### All this is still copied from the tutorial for testing! - Not Mine ###


# Define the model

# Train model 

# Test model

### Produce Detector ###