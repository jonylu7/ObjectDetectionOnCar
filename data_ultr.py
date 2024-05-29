from roboflow import Roboflow

rf = Roboflow(api_key="YXeXgzKhCVUNBnZhK4KH")
project = rf.workspace("manas-mtp").project("mtp-vehicles")
version = project.version(27)
dataset = version.download("yolov5")
