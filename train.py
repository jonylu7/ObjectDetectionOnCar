from ultralytics import YOLO
import cv2
import torch


def valBest():
    model = YOLO("model/yolov5n_best.pt")
    model.export(format="engine")
    # metrics = model.val()
    # print(metrics.box.map50)


def train():
    model = YOLO("model/yolov5n.pt")
    # torch.cuda.empty_cache()
    model.train(data="data.yaml", epochs=100, imgsz=860, device="cuda:1", save=True,
                plots=True)


if __name__ == '__main__':  # Prevent recursive subprocess creation

    # model = YOLOv1ResNet().to(device)
    # jonylu7: load pretrained and newest yolo model

    # train()
    valBest()
