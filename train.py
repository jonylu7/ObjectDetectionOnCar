from ultralytics import YOLO

def loadVideoAndExport():
    print()

def valBest():
    model=YOLO("model/yolov5m_map50_912.pt")
    metrics = model.val()
    print(metrics.box.map50)

def train():
    model = YOLO("model/yolov5m.pt")

    results = model.train(data="data.yaml", epochs=100, imgsz=860, device="cuda", save=True,
                          plots=True)

if __name__ == '__main__':  # Prevent recursive subprocess creation

    # model = YOLOv1ResNet().to(device)
    # jonylu7: load pretrained and newest yolo model

    #train()
    valBest()
