from qoa4ml.collector.amqp_collector import Amqp_Collector
from qoa4ml.connector.amqp_connector import Amqp_Connector
from qoa4ml.reports import Qoa_Client
from qoa4ml.utils import load_config, get_sys_cpu, get_sys_mem
import PIL.Image as Image
import io, cv2, time, json, requests, threading
import numpy as np
from datetime import datetime
from yolov8.yolov8 import Yolo8
from yolov5.yolov5 import Yolo5
import os

url = "http://127.0.0.1:8000/"


def enhance_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_im = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return enhanced_im


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Basic_collector:
    def __init__(self, collector_config: dict, host_object: object = None):
        self.amqp_collector = Amqp_Collector(collector_config, self)
        # self.qoa_client = Qoa_Client(client_config, qoa_config)
        # metrics = load_config('./metrics.json')
        # self.qoa_client.add_metric(metrics)
        self.time = 0
        self.confidence = 0
        self.request_served = 0
        self.timestamp = time.time()

    def message_processing(self, ch, method, props, body):
        start_time = time.time()
        data = json.loads(body)
        image = bytes.fromhex(data["file"])
        result = requests.post(url, files={"image": image})
        result = result.json()
        respond = {}
        try:
            if result is not None:
                respond["prediction"] = result["prediction"]
                respond["image"] = result["image"]
                respond["true_label"] = data["true_label"]
                respond["request_time"] = data["request_time"]
                self.confidence = respond["prediction"]["aggregated"][0]["object_0"][
                    "confidence"
                ]
        except:
            print(result["prediction"])

        print("Server respond time: {}".format(time.time() - start_time))
        self.request_served += 1
        self.timestamp = time.time()
        print("Number of request served: {}".format(self.request_served))
        print("Inference confidence: {}".format(self.confidence))
        print("Time stamp: {}\n".format(datetime.fromtimestamp(self.timestamp)))
        self.amqp_connector.send_data(
            json.dumps(respond, cls=NumpyEncoder), props.correlation_id
        )


server_collector_config = load_config("./server_collector.json")
client_collector_config = load_config("./client_collector.json")
server_collector = Basic_collector(server_collector_config)
client_collector = Basic_collector(client_collector_config)
server_thread = threading.Thread(target=server_collector.amqp_collector.start)
server_thread.start()
client_thread = threading.Thread(target=client_collector.amqp_collector.start)
client_thread.start()
