from ultralytics import YOLO
import cv2


def loadVideoAndExport(sorcefilepath, exportfilepath, frame_width=1280, frame_height=720):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(exportfilepath, fourcc, 20, (frame_width, frame_height))
    # Load the YOLOv8 model
    model = YOLO("model/yolov5m_map50_912.pt")

    # Open the video file
    cap = cv2.VideoCapture(sorcefilepath)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def valBest():
    model = YOLO("model/yolov5m_map50_912.pt")
    metrics = model.val()
    print(metrics.box.map50)


def train():
    model = YOLO("model/yolov5m.pt")

    results = model.train(data="data.yaml", epochs=100, imgsz=860, device="cuda", save=True,
                          plots=True)


if __name__ == '__main__':  # Prevent recursive subprocess creation

    # train()
    # valBest()
    loadVideoAndExport("dashcam/dashcam-2.mp4", "result_videos/dcam_result-2.mp4", 1280, 720)
