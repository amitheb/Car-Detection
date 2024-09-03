YOLO Car Detection
This project uses the YOLO (You Only Look Once) object detection model to detect cars in a video and output relevant metrics. The code uses OpenCV for video processing and the YOLO model for object detection.
**Note : Since weight file is more than 100 MB. it is not possible to uplode or commit to git repo.**
Features
YOLO Model Loading: Loads a YOLO model with specified configuration and weights.
Class Names Loading: Loads class names from a specified file.
Video Processing: Processes the input video to detect cars and saves the result to an output video file.
Metrics Calculation: Computes and saves metrics such as the total number of detected cars, true positives, false positives, and precision.
Output: Saves the processed video and metrics to specified files.
Requirements
Python 3.x
OpenCV (cv2)
NumPy
You can install the required Python packages using pip:

bash
Copy code
pip install opencv-python numpy
File Structure
cfg/yolov3.cfg: YOLO configuration file.
weights/yolov3.weights: YOLO weights file.
data/coco.names: File containing class names.
video/target_video.mp4: Input video file.
output/output_video.mp4: Output video file with detections.
output/car_count.txt: Text file with the total number of detected cars.
output/metrics.txt: Text file with detection metrics including precision.
Usage
Set up Paths: Ensure that the paths to the YOLO configuration file, weights, class names file, and input video file are correctly specified in the main function.

Run the Script: Execute the script using Python:

bash
Copy code
python your_script_name.py
Check Outputs:

The processed video with detected cars will be saved to output/output_video.mp4.
Metrics and car count will be saved to output/metrics.txt and output/car_count.txt, respectively.
Troubleshooting
File Not Found Errors: Make sure that all required files exist in their specified paths.
OpenCV Errors: Ensure that OpenCV is installed correctly and that the video file is accessible.
Unexpected Errors: Check the console output for details on the error and ensure that the script is run in an environment where all dependencies are satisfied.
Contributing
Feel free to submit issues or pull requests to improve the project.

License
This project is licensed under the MIT License. See the LICENSE file for details.
