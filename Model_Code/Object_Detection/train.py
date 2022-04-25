### NOTES ###
# This needs a refactor! 
# - Super helpful tutorial: https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5
# - Create a training timer object? 
### NOTES ###

### Training Script ###

### External Imports ###
import albumentations as A # For data augmentations that include bounding boxes
from albumentations.pytorch import ToTensorV2
import json
import torch # Get more specific things
import time  
from torch.utils.data import DataLoader
from torchvision import transforms as T
### External Imports ###

### Local Imports ### 
from produce_dataset import ProduceDataset  
from csv_logger import CSVLogger
from produce_detector import get_model
from visualization import visualize_predictions
### Local Imports ### 

## Temp ##
import time
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
## Temp ##

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
## JSON was probably overkill, how can I make this more readable ##

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

def collate_fn(batch):
    return tuple(zip(*batch))

# Time should be gathered here? (make t_finish optional!!!)
def time_elapsed(t_finish, t_start):
        t_elapsed = time.gmtime((t_finish - t_start))
        return (time.strftime("%H:%M:%S", t_elapsed))

# A stack of transforms for data augmentation
# Train should be a boolean
def get_transform(train):
    if train:
        train_transform = A.Compose(
            [
                A.ToFloat(),
                A.PadIfNeeded(min_height=800, min_width=800, border_mode=0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(),
                #A.ColorJitter(p=0.2),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']) # pascal_voc is the same as the pytorch convention
        )


    return train_transform

def main():

    kiwi_banana = ProduceDataset(root="assets/datasets/banana_kiwi/images/", 
                                   annotations="assets/datasets/banana_kiwi/banana_kiwi.json",
                                   transforms=get_transform(train=True))

    new_dataset = ProduceDataset(root=root_path, 
                                   annotations=annotations_path,
                                   transforms=get_transform(train=True))
    
    full_dataset = torch.utils.data.ConcatDataset([kiwi_banana, new_dataset])

    train_size = int(0.5 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    val_size = int(0.2 * test_size)
    test_size -= int(0.2 * test_size)

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)


    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    stats = CSVLogger('train_stats00.csv', ["Epoch", "Time", "Loss"]) # These should also go in params.json

    model = get_model(num_classes)
    model.to(device)

    visualize_predictions(train_dataloader)

    # Make this only one epoch and bring main loop out here
    train(model, train_dataloader, test_dataloader, stats)

    visualize_predictions(test_dataloader, model)
    test(model, test_dataloader, "cpu") 

    torch.save(model.state_dict(), "detector.pt")


# Train the model
def train(model, train_dataloader, test_dataloader, stats):
        
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, 
                                lr=learning_rate, 
                                momentum=momentum
                                #weight_decay=weight_decay
                                )

    # Add lr scheduler to params.json 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    len_dataloader = len(train_dataloader)

    print("Starting Training!")
    t_start = time.time()
    t_last_epoch = t_start

    for epoch in range(num_epochs):
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

        # Test and visualize every 10 epochs (Broken)
        # if((epoch + 1) % 10 == 0):
        #     test(model, test_dataloader, "cpu") 
        #     visualize_predictions(test_dataloader, model)

    t_finish = time.time()
    print("Time training: " + time_elapsed(t_finish, t_start))

## These should belong in their own modules ##
# Compare the trained model to test dataset
# Taken from: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def test(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator 
# Taken from: https://github.com/pytorch/vision/blob/main/references/detection/engine.py

main()

### Training Script ###