from qoa4ml.collector.amqp_collector import Amqp_Collector
from qoa4ml.connector.amqp_connector import Amqp_Connector
from qoa4ml.utils import load_config, get_sys_cpu, get_sys_mem
import PIL.Image as Image
import io,cv2, time, json, requests
import numpy as np
from yolov8.yolov8 import Yolo8
from yolov5.yolov5 import Yolo5
import os
url = 'http://127.0.0.1:8000/'
def enhance_image(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    enhanced_im = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return enhanced_im
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
class Basic_collector():
    def __init__(self, collector_config: dict, connector_config: dict, host_object: object=None):
        self.amqp_collector = Amqp_Collector(collector_config, self)
        self.amqp_connector = Amqp_Connector(connector_config)
        self.time = 0
        self.first_request = True
        self.request_handled = 0
    def message_processing(self, ch, method, props, body):
        if self.first_request:
            self.first_request = False
            self.time = time.time()
        start_time = time.time()
        result = requests.post(url, files={'image': body}) 
        result = result.json()
        respond = {}
        if result is not None:
            respond['prediction'] = result['prediction']
            respond['image']      = result['image']
        print("Server respond time: {}".format(time.time() - start_time))
        self.request_handled += 1
        print("Throughtput: {}\n".format((time.time() - self.time)/self.request_handled))
        self.amqp_connector.send_data(json.dumps(respond, cls=NumpyEncoder), props.correlation_id)

collector_config = load_config('./server_collector.json')
connector_config = load_config('./server_connector.json')
basic_collector = Basic_collector(collector_config, connector_config)
basic_collector.amqp_collector.start()
