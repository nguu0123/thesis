from qoa4ml.collector.amqp_collector import Amqp_Collector
from qoa4ml.connector.amqp_connector import Amqp_Connector
from qoa4ml.utils import load_config
import PIL.Image as Image
import io,cv2, time, json
import numpy as np
from yolov8.yolov8 import Yolo8
from yolov5.yolov5 import Yolo5
import os
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
        self.yolov5 = Yolo5("yolov5s")
    def message_processing(self, ch, method, props, body):
        start_time = time.time()
        image = Image.open(io.BytesIO(body))
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        en_im = enhance_image(open_cv_image)
        result = self.yolov5.yolov5_inference(en_im)
        respond = {}
        if result is not None:
            respond['prediction'] = result[0]
            respond['image']      = result[1]
        print("Server respond time: {}".format(time.time() - start_time))
        self.amqp_connector.send_data(json.dumps(respond, cls=NumpyEncoder), props.correlation_id)

collector_config = load_config('./server_collector.json')
connector_config = load_config('./server_connector.json')
basic_collector = Basic_collector(collector_config, connector_config)
basic_collector.amqp_collector.start()
