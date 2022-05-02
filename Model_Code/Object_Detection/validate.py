### External Imports ###
import albumentations as A # For data augmentations that include bounding boxes
from albumentations.pytorch import ToTensorV2
import time  
import json
from torch.utils.data import DataLoader
from torchvision import transforms as T
### External Imports ###

### Local Imports ### 
from produce_dataset import ProduceDataset  
from csv_logger import CSVLogger
from produce_detector import get_model
### Local Imports ### 

import utils
import torchvision
import platform
import torch
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# A stack of transforms for data augmentation
# Train should be a boolean
def get_transform(train):
    if train:
        train_transform = A.Compose(
            [
                A.ToFloat(),
                A.PadIfNeeded(min_height=800, min_width=800, border_mode=0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(),
                #A.ColorJitter(p=0.2),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']) # pascal_voc is the same as the pytorch convention
        )

    return train_transform

def collate_fn(batch):
    return tuple(zip(*batch))

def test(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox"])

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

def validate(): 

    params_json = open("Model_Code/Object_Detection/params.json", "r")
    params = json.load(params_json)

    # Dataset Params
    root_path = params["dataset_params"]["root_path"]
    annotations_path = params["dataset_params"]["annotations_path"]

    # Training Dataloader Params 
    train_batch_size = params["training_dataloader_params"]["train_batch_size"]
    shuffle = params["training_dataloader_params"]["shuffle"]
    num_workers = params["training_dataloader_params"]["num_workers"]

    kiwi_banana = ProduceDataset(root="assets/datasets/banana_kiwi/images/", 
                                   annotations="assets/datasets/banana_kiwi/banana_kiwi.json",
                                   transforms=get_transform(train=True))

    new_dataset = ProduceDataset(root=root_path, 
                                   annotations=annotations_path,
                                   transforms=get_transform(train=True))

    final_dataset = ProduceDataset(root="assets/datasets/fruit_train02/images/", 
                                   annotations="assets/datasets/fruit_train02/fruit_train03.json",
                                   transforms=get_transform(train=True))

    lime_dataset = ProduceDataset(root="/home/colin/SeniorProject/assets/datasets/fruit_train03/images/", 
                                   annotations="/home/colin/SeniorProject/assets/datasets/fruit_train03/extra_lime.json",
                                   transforms=get_transform(train=True))
                                   
                                   
    full_dataset = torch.utils.data.ConcatDataset([kiwi_banana, new_dataset, final_dataset, lime_dataset])

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    test_size = int(0.2 * train_size)
    train_size -= int(0.2 * train_size)

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))

    val_dataloader = DataLoader(val_dataset, 
                                  batch_size=train_batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 9) 

    plt = platform.system()
    if( plt == "Windows"):
        model.load_state_dict(torch.load('C:/Users/Colin/Desktop/SeniorProject/detector.pt'))
    elif( plt == "Linux"):
        model.load_state_dict(torch.load('/home/colin/SeniorProject/Server_Code/Tests/detector.pt'))

    model.eval() 
    model.to(device)

    test(model, val_dataloader, device)

validate()