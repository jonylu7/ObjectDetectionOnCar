import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from data import BoundingBoxDataset
from loss import SumSquaredErrorLoss
from model import *
from ultralytics import YOLO

if __name__ == '__main__':  # Prevent recursive subprocess creation

    # model = YOLOv1ResNet().to(device)
    # jonylu7: load pretrained and newest yolo model
    model = YOLO("model/yolov5s.pt")

    results = model.train(data="data.yaml", epochs=100, imgsz=640, device="mps", save=True,
                          plots=True)
