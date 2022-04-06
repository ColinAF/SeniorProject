### NOTES ###
# -Add a module that keeps track of hyperparams and paths 
### NOTES ###

### Produce Detector ###

### External Imports ###
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt 
### External Imports ###

### Local Imports ### 
from data_utils import DATASETS_PATH # Find a smarter way to go about this? 
from produce_dataset import ProduceDataset
### Local Imports ### 


# Potentially put each step in its own file! 

# Load custom dataset 
train_dataset = ProduceDataset(root=DATASETS_PATH,transforms=None)

# Put the data in a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Put all this in hyperparams

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
# image shape is [batch_size, 3 (due to RGB), height, width]
print(train_features.shape)
img = transforms.ToPILImage()(train_features[0])
plt.imshow(img)
plt.show()
#print(train_labels)



# Define the model

# Train model 

# Test model

### Produce Detector ###