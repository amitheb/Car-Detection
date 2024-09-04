import cv2
import numpy as np
import sys
import os
import time

class YOLOCarDetector:
    def __init__(self, cfg_path, weights_path, coco_names_path):
        self.net, self.output_layers = self.load_yolo_model(cfg_path, weights_path)
        self.classes = self.load_class_names(coco_names_path)
        self.car_count = 0
        self.true_positives = 0
        self.false_positives = 0

    def load_yolo_model(self, cfg_path, weights_path):
        try:
            net = cv2.dnn.readNet(weights_path, cfg_path)
            layer_names = net.getLayerNames()
            unconnected_layers = net.getUnconnectedOutLayers()
            output_layers = [layer_names[i - 1] if isinstance(i, int) else layer_names[i[0] - 1] for i in unconnected_layers]
            return net, output_layers
        except cv2.error as e:
            sys.exit(f"OpenCV error loading YOLO model: {e}")
        except Exception as e:
            sys.exit(f"Unexpected error loading YOLO model: {e}")

    def load_class_names(self, coco_names_path):
        try:
            with open(coco_names_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            sys.exit(f"Class names file not found: {coco_names_path}")
        except Exception as e:
            sys.exit(f"Error loading class names: {e}")

    def process_frame(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for output in outs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        self.annotate_frame(frame, indexes, boxes, class_ids, confidences)

    def annotate_frame(self, frame, indexes, boxes, class_ids, confidences):
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            if label == "car":
                self.car_count += 1
                self.true_positives += 1
                color = (0, 255, 0)
            else:
                self.false_positives += 1
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def calculate_precision(self):
        total_predictions = self.true_positives + self.false_positives
        return self.true_positives / total_predictions if total_predictions > 0 else 0

    def save_metrics(self, count_file_path, metrics_file_path):
        precision = self.calculate_precision()
        with open(count_file_path, 'w') as f:
            f.write(f"Total cars detected: {self.car_count}\n")

        with open(metrics_file_path, 'w') as f:
            f.write(f"True Positives (Cars): {self.true_positives}\n")
            f.write(f"False Positives (Other classes): {self.false_positives}\n")
            f.write(f"Precision: {precision:.2f}\n")

        print(f"Total cars detected: {self.car_count}")
        print(f"Precision: {precision:.2f}")

    def process_video(self, video_path, output_path, max_duration=60):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            sys.exit("Error: Could not open video file.")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if time.time() - start_time > max_duration:
                print("Reached maximum video duration of 1 minute.")
                break

            self.process_frame(frame)
            out.write(frame)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
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
        count_file_path = "output/car_count.txt"
        metrics_file_path = "output/metrics.txt"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(count_file_path), exist_ok=True)

        # Ensure all paths exist
        for path in [cfg_path, weights_path, coco_names_path, video_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")

        # Initialize YOLO detector
        yolo_detector = YOLOCarDetector(cfg_path, weights_path, coco_names_path)

        # Process the video
        yolo_detector.process_video(video_path, output_path)

        # Save metrics
        yolo_detector.save_metrics(count_file_path, metrics_file_path)

        # Keep the frame window active for a few seconds before closing
        time.sleep(3)
        cv2.destroyAllWindows()

    except Exception as e:
        sys.exit(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()