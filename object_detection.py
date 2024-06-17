import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the TensorFlow model from the specified absolute path
model_path = r'C:\Users\Lenovo\PycharmProjects\pythonProject3\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\saved_model'
model = tf.saved_model.load(model_path)

# Define the categories based on the COCO dataset labels (Extended to include more items)
category_index = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    44: {'id': 44, 'name': 'bottle'},
    62: {'id': 62, 'name': 'chair'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'}
}

def detect_and_announce(frame):
    # Convert frame to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(frame_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform detection
    detections = model(input_tensor)

    # Analyze the detection outputs
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)

    # Announce all detected objects with modified confidence
    announced = []
    for i, score in enumerate(detection_scores):
        if score > 0.3:  # Modified confidence threshold
            class_id = detection_classes[i]
            class_name = category_index.get(class_id, {}).get('name', 'Unknown')
            if class_name not in announced:  # Prevent repeating the same announcement
                engine.say(f"Detected {class_name}")
                engine.runAndWait()
                announced.append(class_name)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break

            frame = detect_and_announce(frame)
            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
