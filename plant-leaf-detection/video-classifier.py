import cv2
import imghdr
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import ultralyticsplus
from ultralyticsplus import YOLO

# Load the YOLO model (replace with your actual model path)
model = YOLO('foduucom/plant-leaf-detection-and-classification')

# Set model parameters for optimal performance (adjust as needed)
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # Class-specific NMS
model.overrides['max_det'] = 1000  # Maximum detections per frame

def process_video(video_path):
    """
    Processes a video using the YOLO model for leaf detection and classification.

    Args:
        video_path (str): Path to the input video file.
    """

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Check if video is opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Get the video's original width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was grabbed successfully
        if not ret:
            print("Could not read frame from stream or file")
            break

        # Preprocess frame based on video format
        frame_preprocessed = preprocess_frame(frame, video_path)

        # Perform leaf detection and classification using the YOLO model
        results = model.predict(frame_preprocessed)

        # Process and visualize detections (optional)
        if results is not None and len(results[0].boxes) > 0:
            # Extract bounding boxes and class labels
            boxes = results[0].boxes
            labels = results[0].names

            # Draw bounding boxes and labels on the frame (optional)
            for box, label in zip(boxes, labels):
                if len(box) == 6:  # Check if there are enough values to unpack
                    x_min, y_min, x_max, y_max, conf, class_id = box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow("Leaf Detection in Video", frame)

        # Exit loop if 'q' key is pressed or the window is closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture resources
    cap.release()
    cv2.destroyAllWindows()

def preprocess_frame(frame, video_path):
    """
    Preprocesses a frame based on the video format.

    Args:
        frame (numpy.ndarray): Input frame to be preprocessed.
        video_path (str): Path to the input video file.

    Returns:
        numpy.ndarray: Preprocessed frame.
    """

    # Determine the file format of the video
    video_format = imghdr.what(video_path)

    # Preprocess frame based on the video format
    if video_format == 'jpeg':
        # Apply specific preprocessing for JPEG format (if needed)
        pass
    elif video_format == 'png':
        # Apply specific preprocessing for PNG format (if needed)
        pass
    elif video_format == 'gif':
        # Apply specific preprocessing for GIF format (if needed)
        pass
    elif video_format == 'bmp':
        # Apply specific preprocessing for BMP format (if needed)
        pass
    elif video_format == 'avi':
        # Apply specific preprocessing for AVI format (if needed)
        pass
    elif video_format == 'mp4':
        # Apply specific preprocessing for MP4 format (if needed)
        pass
    else:
        # Unsupported video format
        print("Unsupported videoformat")
        return None

    # Return the preprocessed frame
    return frame

# Create a GUI window to choose the video file
root = tk.Tk()
root.withdraw()

# Prompt the user to select a video file
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.jpeg;*.png;*.gif;*.bmp")])

# Process the selected video file
if video_path:
    # Create a separate thread for video processing
    video_thread = threading.Thread(target=process_video, args=(video_path,))
    video_thread.start()

    # Run the main Tkinter event loop for the UI
    root.mainloop()