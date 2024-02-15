# AI-dogs-trap-v2

import RPi.GPIO as GPIO
import time
import cv2
import numpy as np

# Define GPIO pins for ULN2003
IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 22

# Define the stepping sequence
SEQUENCE = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
]

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)

def set_step(w1, w2, w3, w4):
    GPIO.output(IN1, w1)
    GPIO.output(IN2, w2)
    GPIO.output(IN3, w3)
    GPIO.output(IN4, w4)

def move(steps, delay):
    for _ in range(steps):
        for step in SEQUENCE:
            set_step(*step)
            time.sleep(delay)

def load_coco_names(names_file):
    with open(names_file, 'r') as f:
        return f.read().strip().split('\n')

def detect_objects(coco_config_path, coco_weights_path, coco_names_path):
    
    # Load COCO model
    net = cv2.dnn.readNetFromDarknet(coco_config_path, coco_weights_path)

    # Explicitly specify the output layer names
    output_layer_names = ['yolo_82', 'yolo_94', 'yolo_106']  # Adjust these names based on your YOLOv3 model structure

    # Load COCO class names
    coco_names = load_coco_names(coco_names_path)
    
    # Open the camera (default camera index is 0, adjust if needed)
    cap = cv2.VideoCapture(0)

    start_time = 0.0
    first_time = 0

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Perform object detection
        # height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layer_names)

        # Loop over the detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.1 and coco_names[class_id] == 'dog':

                    label = f"{coco_names[class_id]}: {confidence:.2f}"
                    print(label)
                    
                    elapsed_time = time.time() - start_time
                    if ((first_time == 0) or (elapsed_time > 30)):
                        first_time = 1
                        start_time = time.time()
                        move(steps=2048, delay=0.001)
                

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

def cleanup():
    GPIO.cleanup()

if __name__ == "__main__":
    try:
        print('Dogs Trap Started...')
        setup()
        
        coco_config_path = "../../app/yolov3.cfg"
        coco_weights_path = "../../app/yolov3.weights"
        coco_names_path = "../../app/coco.names"
        detect_objects(coco_config_path, coco_weights_path, coco_names_path)
        # Rotate the stepper motor 2048 steps (full revolution) with a delay of 0.005 seconds between steps
        # move(steps=2048, delay=0.001)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


