### NOTES ###
# - Add a module that keeps track of hyperparams and paths in json
# - Add more data transforms!!! 
# - Files for {train,test/validate,model,visualizations}
# - Super helpful tutorial: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
# - Export timeings to a file!
# - Consider an adaptive learning rate
# - Visualizations for trained model!! 
### NOTES ###

### Training Script ###

### External Imports ###
import torch # Get more specific things
import time  
from torch.utils.data import DataLoader
from torchvision import transforms
### External Imports ###

### Local Imports ### 
from data_utils import DATASETS_PATH # Put pathnames and hyperparams in JSON 
from produce_dataset import ProduceDataset  
from produce_detector import get_model
### Local Imports ### 

# Dataset Params
root_path = DATASETS_PATH 
annotations_path = "assets/datasets/fruit_test/annotations.json"

# Training Dataloader Params 
train_batch_size = 1
shuffle = True 
num_workers = 4 

# Optamizer Params
learning_rate = 0.005
momentum = 0.9
weight_decay = 0.0005

# Model Params
num_classes = 3

# Training Params 
num_epochs = 10

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu') 

def collate_fn(batch):
    return tuple(zip(*batch))


def time_elapsed(t_finish, t_start):
        t_elapsed = time.gmtime((t_finish - t_start))
        return (time.strftime("%H:%M:%S", t_elapsed))


def main():
    # Load custom dataset 
    train_dataset = ProduceDataset(root=root_path, 
                                annotations=annotations_path,
                                transforms=transforms.ToTensor())
    # Add Data Transforms
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=train_batch_size, 
                                shuffle=shuffle, 
                                num_workers=num_workers,
                                collate_fn=collate_fn)

    train(train_dataloader)

def train(train_dataloader):
    model = get_model(num_classes)
    model.to(device)
        
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, 
                                lr=learning_rate, 
                                momentum=momentum, 
                                weight_decay=weight_decay)

    len_dataloader = len(train_dataloader)

    print("Starting Training!")
    t_start = time.time()
    t_last_epoch = t_start

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()

        t_epoch = time.time()
        t_last_epoch = t_epoch

        i = 0 

        for images, annotations in train_dataloader :
            images = list(image.to(device) for image in images)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(images, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            i+=1

            print(f'Epoch: {epoch+1} Iteration: {i}/{len_dataloader}, Loss: {losses}')
        
        print("Epoch: " + 
              str(epoch+1) + 
              " Time in epoch: " + 
              time_elapsed(t_epoch, t_last_epoch))

    t_finish = time.time()
    print("Time training: " + time_elapsed(t_finish, t_start))

main()
### Training Script ###