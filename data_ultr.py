from roboflow import Roboflow

rf = Roboflow(api_key="YXeXgzKhCVUNBnZhK4KH")
project = rf.workspace("vehicle-counter").project("vehicle-counter-5zkt5")
version = project.version(2)
dataset = version.download("yolov5")
