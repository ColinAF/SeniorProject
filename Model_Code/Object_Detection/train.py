### NOTES ###
# - Files for {train,test/validate,model,visualizations}
# - Super helpful tutorial: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
# - Visualizations for trained model!! 
# - Add test code
# - More Data Augmentation!!!
# - Create a training timer object? 
### NOTES ###

### Training Script ###

### External Imports ###
import json
from matplotlib import transforms 
import torch # Get more specific things
import time  
from torch.utils.data import DataLoader
from torchvision import transforms as T
### External Imports ###

### Local Imports ### 
from produce_dataset import ProduceDataset  
from csv_logger import CSVLogger
from produce_detector import get_model
### Local Imports ### 

## JSON was probably overkill, make this more readable ##
params_json = open("Model_Code/Object_Detection/params.json", "r")
params = json.load(params_json)

# Dataset Params
root_path = params["dataset_params"]["root_path"]
annotations_path = params["dataset_params"]["annotations_path"]

# Training Dataloader Params 
train_batch_size = params["training_dataloader_params"]["train_batch_size"]
shuffle = params["training_dataloader_params"]["shuffle"]
num_workers = params["training_dataloader_params"]["num_workers"]

# Optimizer Params
learning_rate = params["optimizer_params"]["learning_rate"]
momentum = params["optimizer_params"]["momentum"]
weight_decay = params["optimizer_params"]["weight_decay"]

# Model Params
num_classes = params["model_params"]["num_classes"]

# Training Params 
num_epochs = params["training_params"]["num_epochs"]
## JSON was probably overkill, make this more readable ##

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def collate_fn(batch):
    return tuple(zip(*batch))

# Time should be gathered here? (make t_finish optional!!!)
def time_elapsed(t_finish, t_start):
        t_elapsed = time.gmtime((t_finish - t_start))
        return (time.strftime("%H:%M:%S", t_elapsed))

# A stack of transforms for data augmentation
# Train should be a boolean
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotation(90))
        #GaussianBlur
        #ColorJitter

    return T.Compose(transforms)

def main():
    train_dataset = ProduceDataset(root=root_path, 
                                   annotations=annotations_path,
                                   transforms=get_transform(train=True))
    # Add Data Transforms
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    stats = CSVLogger('train_stats00.csv', ["Epoch", "Time", "Loss"]) # These should also go in params.json

    train(train_dataloader, stats)
    test() 

# Train the model
def train(train_dataloader, stats):
    model = get_model(num_classes)
    model.to(device)
        
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, 
                                lr=learning_rate, 
                                momentum=momentum, 
                                weight_decay=weight_decay)

    # Add lr scheduler to params.json 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    len_dataloader = len(train_dataloader)

    print("Starting Training!")
    t_start = time.time()
    t_last_epoch = t_start

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()

        i = 0 

        for images, annotations in train_dataloader :
            images = list(image.to(device) for image in images)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(images, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            i+=1

            print(f'Epoch: {epoch+1} Iteration: {i}/{len_dataloader}, Loss: {losses}')
            stats.log([(epoch+1), (time_elapsed(time.time(), t_start)), f'{losses}'])
        
        t_epoch = time.time()
        print("Epoch: " + str(epoch+1) + " Time in epoch: " + time_elapsed(t_epoch, t_last_epoch))       
        t_last_epoch = t_epoch

    t_finish = time.time()
    print("Time training: " + time_elapsed(t_finish, t_start))

## These should belong in their own modules ##
# Compare the trained model to test dataset
def test():
    pass

# Compare the trained model to validation dataset
def validate():
    pass
## These should belong in their own modules ##

main()

### Training Script ###