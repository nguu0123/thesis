from ultralytics import YOLO
from PIL import Image
import cv2, yaml, os
import pandas as pd
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors


def not_approximate(a, b):
    if abs(a - b) < 10:
        return False
    else:
        return True


def extract_dict(dict, keys):
    result = {}
    for key in keys:
        result[key] = dict[key]
    return result


def compare_box(box1, box2):
    for key in box1:
        if not_approximate(box1[key], box2[key]):
            return False
    return True


path = os.path.dirname(os.path.abspath(__file__)) + "/class.yml"
with open(path, "r") as f:
    names = yaml.safe_load(f)


class Yolo8(object):
    def __init__(self, param=None):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.names = names
        self.param = param if param is not None else "yolov8n"
        self.model = YOLO(self.path + "/model/" + self.param + ".pt")

    def convert_results(self, results, annotator):
        prediction_list = []
        for result in results:
            # convert detection to numpy array
            numpy_result = result.cpu().boxes.numpy().data
            # Cast to pandas DataFrame
            prediction = pd.DataFrame(
                numpy_result,
                columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"],
            )
            # Map to class names
            prediction["name"] = prediction["class"].apply(lambda x: names["names"][x])
            # Label object and annotate
            for index, row in prediction.iterrows():
                xyxy = row.values.flatten().tolist()[:-2]
                c = int(row["class"])
                label = row["name"] + ":" + str(row["confidence"])
                annotator.box_label(xyxy, label, color=colors(c, True))

            # Conver prediction to dictionary to store in DB

            pre_dict = prediction.to_dict("index")
            prediction = []
            key_list = list(pre_dict.keys())
            val_list = list(pre_dict.values())
            object_count = 0
            while key_list:
                pre_obj = [val_list[0]]
                box1 = extract_dict(val_list[0], ["xmin", "ymin", "xmax", "ymax"])
                for i in range(1, len(key_list)):
                    box2 = extract_dict(val_list[i], ["xmin", "ymin", "xmax", "ymax"])
                    if compare_box(box1, box2):
                        pre_obj.append(val_list[i])
                        pre_dict.pop(key_list[i])
                detect_obj = {f"object_{object_count}": pre_obj}
                pre_dict.pop(key_list[0])
                key_list = list(pre_dict.keys())
                val_list = list(pre_dict.values())
                object_count += 1
                prediction.append(detect_obj)
            return {self.param: prediction}, annotator.result()

    def yolov8_inference(self, image):
        annotator = Annotator(image, line_width=1)
        # Inference
        results = self.model(image, verbose=0)
        return self.convert_results(results, annotator)


# Images

# img = cv2.imread('../img/dog.jpg')  # or file, Path, PIL, OpenCV, numpy, list
# prediction, pre_img = yolov8_inference(img)
# print(prediction)
# cv2.imshow("lable",pre_img)
# cv2.waitKey(0)
