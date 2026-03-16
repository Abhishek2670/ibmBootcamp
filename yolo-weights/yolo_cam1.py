# import lib
from ultralytics import YOLO
import cv2
import math

# staRT webcamp

cap = cv2.VideoCapture(0)
cap.set(1,640)
cap.set(1,480)

# load the yolo model

model = YOLO("yolo-weights/yolo11n.pt")

classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush','pen','fan','bowl','mug','mouse','keyboard','cell phone','laptop','monitor','printer',   'speaker','router','table lamp','backpack','handbag','suitcase' ,    'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush','stool','ladder','projector','projector screen','whiteboard','trash can','bottle','mug','bowl','plate','camera','flashlight','tripod','binoculars','watch','ring','necklace','bracelet','earrings']

# infinite loop to continuously get images from the webcam
while True:
    success, img = cap.read()

    # make detections
    results = model(img,stream=True)

    # post process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(63, 150, 142),2)

            # confidence
            conf = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            cv2.putText(img,f'{currentClass} {conf}',(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)