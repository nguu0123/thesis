import argparse
import json
from math import acos
import os
import os.path
import random
import time
import uuid
import csv
from csv import writer
from datetime import datetime
from functools import reduce

import aggregation
import cv2
import numpy as np
import pandas as pd
import pymongo
import qoa4ml.utils as qoa_utils
import ray
import requests
from fastapi import FastAPI
from qoa4ml.reports import Qoa_Client
from qoa4ml.utils import load_config
from ray import serve
from ray.serve.deployment_graph import InputNode
from ray.serve.drivers import DAGDriver
from ray.serve.handle import RayServeDeploymentHandle
from ray.serve.http_adapters import json_request
from starlette.requests import Request
from yolov5.yolov5 import Yolo5
from yolov8.yolov8 import Yolo8
from prov.prov_function import captureInputData, capture, captureModel 

@ray.remote
@capture(activityType='preprocessing')
def enhance_image(data):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_im = cv2.filter2D(src=data.data, ddepth=-1, kernel=kernel)
    return enhanced_im


@ray.remote
@capture(activityType='ensemble')
def mean_aggregate(predictions):
    agg_prediction = aggregation.agg_mean(list(map(lambda x: x.data, predictions)))
    report = {}
    objectDetected = []
    #for curObject in agg_prediction[0]:
    #    objectDetected.append((agg_prediction[0][curObject]["name"], agg_prediction[0][curObject]["confidence"]))
    #report["object detected"] = objectDetected
    return {"prediction": agg_prediction, "QoA" : report}

@ray.remote
@capture(activityType='ensemble')
def max_aggregate(predictions):
    agg_prediction = aggregation.agg_max(reduce(lambda a, b: a.data | b.data, predictions))
    report = {}
    objectDetected = []
    #for curObject in agg_prediction[0]:
    #    objectDetected.append((agg_prediction[0][curObject]["name"], agg_prediction[0][curObject]["confidence"]))
    #report["object detected"] = objectDetected
    return {"prediction": agg_prediction, "QoA" : report}


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 3,
        "upscale_delay_s": 10
    }
)
class Yolo8Inference:
    @captureModel
    def __init__(self, param):
        self.model = Yolo8(param)
        self.param = param

    @capture(activityType='predict')
    def predict(self, data):
        prediction, pre_img = self.model.yolov8_inference(data.data)
        return {"prediction": prediction, "image": pre_img, "model_name": self.param, "QoA": self.extractQoA(prediction)}

    def extractQoA(self, prediction):
        report = {}
        objectDetected = []
        for model in prediction.keys():
            if prediction[model]:
                data = prediction[model] 
                object_num = 0
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    objectDetected.append((cur_object["name"], cur_object["confidence"]))    
                    object_num += 1
        report["object detected"] = objectDetected
        return report

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 3,
        "upscale_delay_s": 10
    }
)
class Yolo5Inference:
    @captureModel
    def __init__(self, param):
        self.model = Yolo5(param)
        self.param = param
        self.inference_time = 0

    @capture(activityType='predict')
    def predict(self, data):
        prediction, pre_img = self.model.yolov5_inference(data.data)
        return {"prediction": prediction, "image": pre_img, "model_name": self.param, "QoA": self.extractQoA(prediction)}

    def extractQoA(self, prediction):
        report = {}
        objectDetected = []
        for model in prediction.keys():
            if prediction[model]:
                data = prediction[model] 
                object_num = 0
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    objectDetected.append((cur_object["name"], cur_object["confidence"]))    
                    object_num += 1
        report["object detected"] = objectDetected
        return report
@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 5,
    }
)
class Ensemble_ML:
    def __init__(self, yolo5, yolo8):
        self.yolo5 = yolo5
        self.yolo8 = yolo8
    async def inference(self, data):
        np_array = np.frombuffer(data, np.uint8)
        im = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        im =  captureInputData(im)
        self.assessDataQuality(data=im)
        en_im = enhance_image.remote(data=im)
        response = {}
        yl5 = await self.yolo5.predict.remote(data=en_im)
        yl8 = await self.yolo8.predict.remote(data=en_im)
        yl5 = ray.get(yl5)
        yl8 = ray.get(yl8)
        agg_pred = await mean_aggregate.remote(predictions = [yl5.getAs("prediction"), yl8.getAs("prediction")] )
        response["prediction"] = {
            "aggregated": agg_pred.data["prediction"], yl5.data["model_name"]: yl5.data["prediction"][yl5.data["model_name"]],
            yl8.data["model_name"]: yl8.data["prediction"][yl8.data["model_name"]],
        }
        response["image"] = yl8.data["image"]
        response['requestId'] = im.requestId
        return response
    @capture(activityType='assessDataQuality')
    def assessDataQuality(self, data):
        image = data.data
        height, width = image.shape[:2]
        resolution = round(width, 2), round(height, 2)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = round(cv2.Laplacian(gray_image, cv2.CV_64F).var(), 2)
        sharpness = round(cv2.Laplacian(gray_image, cv2.CV_64F).var(), 2)
        brightness = round(np.mean(gray_image), 2)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = round(np.mean(hsv_image[:, :, 1]), 2)

        noise_mask = cv2.inRange(gray_image, 0, 30) + cv2.inRange(gray_image, 225, 255)
        noise = np.mean(noise_mask) / 255 * 100

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        blur = np.var(laplacian)
        metrics = {
            'resolution': resolution,
            'contrast': contrast,
            'sharpness': sharpness,
            'brightness': brightness,
            'saturation': saturation,
            'noise': noise,
            'blur': blur
        }
        return metrics
@serve.deployment
class MostBasicIngress:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.respond_time = 0
        self.request_served = 0
        self.log_file_name = "server_log.csv"
        self.models_log_comlumns = [
            "Image ID",
            "Model",
            "Object detected",
            "Confidence",
        ]
        self.server_log_columns = ["Image ID", "Response time"]
    async def __call__(self, request: Request):
        start_time = time.time()
        time_stamp = datetime.fromtimestamp(time.time())
        files = await request.form()
        data = await files["data"].read()
        request_id = str(uuid.uuid4())
        image_id = await files["file name"].read()
        response = await self.ensemble.inference.remote(data)
        response = ray.get(response)
        return response

    def monitor(self, start_time, times_stamp, inference_result, image_id):
        for model in inference_result.keys():
            if inference_result[model]:
                data = inference_result[model] 
                object_num = 0
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    self.append_to_log(
                        [
                            image_id,
                            model,
                            cur_object["name"],
                            cur_object["confidence"],
                        ],
                        "models_log",
                    )
                    object_num += 1
        self.append_to_log([image_id, time.time() - start_time], 'server_log')
        self.request_served += 1

    def append_to_log(self, data, file_name):
        path = "../logs/{}.csv".format(file_name)
        if not os.path.exists(path):
            with open(path, "w") as f:
                writer = csv.writer(f)
                if file_name == "server_log":
                    writer.writerow(self.server_log_columns)
                elif file_name == "models_log":
                    writer.writerow(self.models_log_comlumns)
        with open(path, "a") as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(data)

models_config = load_config("../config/models.json")
yolo5 = Yolo5Inference.bind(param=models_config['yolo5'])
yolo8 = Yolo8Inference.bind(param=models_config['yolo8'])
ensemble = Ensemble_ML.bind(yolo5, yolo8)
server = MostBasicIngress.bind(ensemble)
