import requests,time, argparse
from starlette.requests import Request
import random
from fastapi import FastAPI
from ray import serve
import pymongo
import json, cv2
import numpy as np
import os, time, uuid, ray
from qoa4ml.reports import Qoa_Client
from qoa4ml.utils import load_config
import qoa4ml.utils as qoa_utils
import aggregation
from yolov8.yolov8 import Yolo8
from yolov5.yolov5 import Yolo5

@ray.remote
def enhance_image(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    enhanced_im = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    print("enhance_image")
    return enhanced_im

@ray.remote
def mean_aggregate(predictions):
    agg_prediction = aggregation.agg_mean(predictions)
    print("mean_agg")
    return agg_prediction

@ray.remote
def max_aggregate(predictions):
    agg_prediction = aggregation.agg_max(predictions)
    print("max_agg")
    return agg_prediction


@serve.deployment
class Yolo8Inference:
    def __init__(self, param):
        self.model = Yolo8(param)

    def predict(self, image):
        prediction, pre_img = self.model.yolov8_inference(image)
        print("yolo8")
        return {"prediction": prediction, "image": pre_img}

@serve.deployment
class Yolo5Inference:
    def __init__(self, param):
        self.model = Yolo5(param)


    def predict(self, image):
        prediction, pre_img = self.model.yolov5_inference(image)
        print("yolo5")
        return {"prediction": prediction, "image": pre_img}

@serve.deployment
class Ensemble_ML:
    def __init__(self,yolo5, yolo8):
        self.yolo5 = yolo5
        self.yolo8 = yolo8
    async def object_detect(self, http_request: Request):
        files = await http_request.form()
        image = await files["image"].read()
        np_array = np.frombuffer(image, np.uint8)
        im = cv2.imdecode(np_array, cv2.IMREAD_COLOR) 
        en_im = enhance_image.remote(im)
        response = {}
        yl5 = await self.yolo5.predict.remote(en_im)
        yl8 = await self.yolo8.predict.remote(en_im)
        yl5 = ray.get(yl5)
        yl8 = ray.get(yl8)
        yl = yl5["prediction"]
        yl.update(yl8["prediction"])
        agg_pred = await max_aggregate.remote(yl)
        response["prediction"] = {"aggregated":agg_pred,"yolo5":yl5["prediction"], "yolo8": yl8["prediction"]}
        response["image"] = yl8["image"]
        return response
@serve.deployment   
class Serving_Edge:
    def __init__(self, ensemble_ml):
        self.ensemble_ml = ensemble_ml  
        client_config = load_config('./client.json')
        connector_config = load_config('./connector.json')
        metric_config = load_config('./metrics.json')
        self.qoa_client = Qoa_Client(client_config, connector_config)
        self.qoa_client.add_metric( metric_config['App-metric'])
        self.metrices = self.qoa_client.get_metric()
    async def __call__(self, http_request: Request):
        start_time = time.time()
        response = await self.ensemble_ml.object_detect.remote(http_request)
        response = ray.get(response)
        response_time = time.time() - start_time   
        self.metrices["Responsetime"].set(response_time)
        self.qoa_client.report(self.metrices["Responsetime"].to_dict())
        return response

yolo5 = Yolo5Inference.bind("yolov5n")
yolo8 = Yolo8Inference.bind("yolov8n")
ensemble = Ensemble_ML.bind(yolo5, yolo8)
serving_edge = Serving_Edge.bind(ensemble)