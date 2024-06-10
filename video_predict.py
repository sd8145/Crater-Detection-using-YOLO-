import cv2
from ultralytics import YOLO

import torch
# Load the YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")

# Input video file path
video_path = "Untitled design.mp4"

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Get video frame dimensions
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Define the codec and create a VideoWriter object to save the output video
output_path = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

with torch.no_grad():
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # Perform object detection using YOLOv8 on the current frame
        results = model(source=frame)
        res_plotted = results[0].plot()

        # Convert the plotted result to BGR format for displaying with OpenCV
        res_bgr = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)

        # Display the frame with detected objects
        cv2.imshow("Video Prediction", res_bgr)


        # Write the frame to the output video
        out.write(res_bgr)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and output objects
video_capture.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()