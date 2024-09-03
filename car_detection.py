import cv2
import numpy as np
import sys
import os
import time

def load_yolo_model(cfg_path, weights_path):
    try:
        net = cv2.dnn.readNet(weights_path, cfg_path)
        layer_names = net.getLayerNames()
        unconnected_layers = net.getUnconnectedOutLayers()
        if len(unconnected_layers.shape) == 1:
            output_layers = [layer_names[i - 1] for i in unconnected_layers]
        else:
            output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
        return net, output_layers
    except cv2.error as e:
        print(f"OpenCV error loading YOLO model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading YOLO model: {e}")
        sys.exit(1)

def load_class_names(coco_names_path):
    try:
        with open(coco_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except FileNotFoundError:
        print(f"Class names file not found: {coco_names_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading class names: {e}")
        sys.exit(1)

def initialize_video(video_path, output_path, frame_width, frame_height):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Could not open video file")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
        return cap, out
    except IOError as e:
        print(f"Error opening video file: {e}")
        sys.exit(1)
    except cv2.error as e:
        print(f"OpenCV error initializing video: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error initializing video: {e}")
        sys.exit(1)

def process_video(cap, out, net, output_layers, classes, count_file_path, metrics_file_path):
    try:
        car_count = 0
        true_positives = 0
        false_positives = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids, confidences, boxes = [], [], []

            for output in outs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.6:  # Confidence threshold for detection
                        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                        w, h = int(detection[2] * width), int(detection[3] * height)
                        x, y = int(center_x - w / 2), int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]

                    if label == "car":
                        car_count += 1
                        true_positives += 1
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = f"{label} {confidence:.2f}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        false_positives += 1

            out.write(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Calculate precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Save car count and precision to a file
        with open(count_file_path, 'w') as f:
            f.write(f"Total cars detected: {car_count}\n")

        with open(metrics_file_path, 'w') as f:
            f.write(f"True Positives (Cars): {true_positives}\n")
            f.write(f"False Positives (Other classes): {false_positives}\n")
            f.write(f"Precision: {precision:.2f}\n")

        print(f"Total cars detected: {car_count}")
        print(f"Precision: {precision:.2f}")

    except cv2.error as e:
        print(f"OpenCV error processing frame: {e}")
    except Exception as e:
        print(f"Unexpected error processing frame: {e}")
    finally:
        if isinstance(cap, cv2.VideoCapture):
            cap.release()
        if isinstance(out, cv2.VideoWriter):
            out.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Paths
        cfg_path = "cfg/yolov3.cfg"
        weights_path = "weights/yolov3.weights"
        coco_names_path = "data/coco.names"
        video_path = "video/target_video.mp4"
        output_path = "output/output_video.mp4"
        count_file_path = "output/car_count.txt"  # File to store the car count
        metrics_file_path = "output/metrics.txt"  # File to store precision and other metrics

        # Ensure output directory exists
        os.makedirs(os.path.dirname(count_file_path), exist_ok=True)

        # Ensure all paths exist
        for path in [cfg_path, weights_path, coco_names_path, video_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        # Load YOLO model and class names
        net, output_layers = load_yolo_model(cfg_path, weights_path)
        classes = load_class_names(coco_names_path)

        # Initialize video processing
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cap.release()

        cap, out = initialize_video(video_path, output_path, frame_width, frame_height)

        # Process video
        process_video(cap, out, net, output_layers, classes, count_file_path, metrics_file_path)

        # Keep the frame window active for a few seconds before closing
        time.sleep(3)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An unexpected error occurred in the main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()