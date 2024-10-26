import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Initialize Detectron2 model configuration
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Load a simple pre-trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detection
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # Load weights
cfg.MODEL.DEVICE = "cpu"  # Use CPU, or "cuda" if you have a compatible GPU

# Create predictor
predictor = DefaultPredictor(cfg)

# Start webcam
cap = cv2.VideoCapture("http://127.0.0.1:8080/video")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    outputs = predictor(frame)

    # Visualize results
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Display output
    cv2.imshow("Webcam", out.get_image()[:, :, ::-1])

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()