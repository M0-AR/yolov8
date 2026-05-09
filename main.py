# https://itnext.io/how-to-explore-and-visualize-ml-data-for-object-detection-in-images-88e074f46361
import pandas as pd
import numpy as np

import fiftyone.zoo as foz

# 01
# download 1000 images from the COCO dataset with persons
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=[
        "detections",
    ],
    classes=["person"],
    max_samples=1000,
    dataset_name="coco-2017-person-1k-validations",
)

# 02
def xywh_to_xyxyn(bbox):
    """convert from xywh to xyxyn format"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


row = []
for i, sample in enumerate(dataset):
    labels = [detection.label for detection in sample.ground_truth.detections]
    bboxs = [
        xywh_to_xyxyn(detection.bounding_box)
        for detection in sample.ground_truth.detections
    ]
    bboxs_persons = [bbox for bbox, label in zip(bboxs, labels) if label == "person"]
    row.append([sample.filepath, labels, bboxs, bboxs_persons])

df = pd.DataFrame(row, columns=["filepath", "categories", "bboxs", "bboxs_persons"])
df["major_category"] = df["categories"].apply(
    lambda x: max(set(x) - set(["person"]), key=x.count)
    if len(set(x)) > 1
    else "only person"
)

# 03
from renumics import spotlight
# spotlight.show(df)

# 04
# spotlight.show(df, embed=["filepath"])


# 05
from ultralytics import YOLO
detection_model = YOLO("yolov8n.pt")

# 06
detections = []
for filepath in df["filepath"].tolist():
    detection = detection_model(filepath)[0]
    detections.append(
        {
            "yolo_bboxs": [np.array(box.xyxyn.tolist())[0] for box in detection.boxes],
            "yolo_conf_persons": np.mean([
                np.array(box.conf.tolist())[0]
                for box in detection.boxes
                if detection.names[int(box.cls)] == "person"
            ]),
            "yolo_bboxs_persons": [
                np.array(box.xyxyn.tolist())[0]
                for box in detection.boxes
                if detection.names[int(box.cls)] == "person"
            ],
            "yolo_categories": np.array(
                [np.array(detection.names[int(box.cls)]) for box in detection.boxes]
            ),
        }
    )
df_yolo = pd.DataFrame(detections)


# 07
df_merged = pd.concat([df, df_yolo], axis=1)
spotlight.show(df_merged, embed=["filepath"])