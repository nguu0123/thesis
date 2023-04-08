import requests, time, argparse, csv
from threading import Thread
import os, random, json, cv2, threading
import numpy as np
from memory_profiler import profile
from qoa4ml.connector.amqp_connector import Amqp_Connector
from qoa4ml.collector.amqp_collector import Amqp_Collector
from qoa4ml.utils import load_config

parser = argparse.ArgumentParser(description="Data processing")
parser.add_argument("--th", help="number concurrent thread", default=1)
parser.add_argument("--sl", help="time sleep", default=-1.0)
args = parser.parse_args()
concurrent = int(args.th)
time_sleep = float(args.sl)
# url = 'http://192.168.1.148:5000/inference_aaltosea'
url = "http://127.0.0.1:8000/"


def sender(num_thread):
    count = 0
    start_time = time.time()
    while time.time() - start_time < 600:
        print("This is thread: ", num_thread, "Starting request: ", count)
        ran_file = random.choice(os.listdir("./image"))
        files = {"image": open("./image/" + ran_file, "rb")}
        response = requests.post(url, files=files)
        prediction = response.json()
        print(prediction["prediction"]["yolo5"])
        print(prediction["prediction"]["yolo8"])
        image = np.asarray((prediction["image"]))

        # Comment this to use multi-thread
        cv2.imshow("lable", image.astype(np.uint8))
        cv2.waitKey(2000)
        time.sleep(10000)

        print("Thread - ", num_thread, " Response:")
        count += 1
        if time_sleep == -1:
            time.sleep(1)
        else:
            time.sleep(time_sleep)


# 1 Thread

class Client:
    def __init__(self, connector_config: dict, log: bool = False):
        # self.amqp_connector = Amqp_Connector(connector_config, log)
        self.correct_detect = 0
        self.request_sent = 0 
        self.request_log_comlums   = ["Image ID", "Response time"]
        self.inference_log_columns = ["Image ID", "Model", "Class", "Correctly detect", "Number of objects detected", 'Width', 
                                      'Height', 'Object width', 'Object height', 'Object to Image percentage']
    def send_message(self):
        ran_class = random.choice(os.listdir("./images/"))
        ran_file = random.choice(os.listdir("./images/{}".format(ran_class)))
        file = open("./images/" + ran_class + "/" + ran_file, "rb").read()
        start_time = time.time()
        response = requests.post(url, files={"data": file, "file name": ran_file})
        prediction = response.json()
        np_array = np.frombuffer(file, np.uint8)
        im = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        height, width, _ = im.shape
        #self.monitor(time.time() - start_time, prediction["prediction"], ran_file, ran_class, height, width)

    def monitor(self, response_time, inference_result, image_id, true_class, image_height, image_width):
        for model in inference_result.keys():
            if inference_result[model]:
                data = inference_result[model] 
                object_num = 0
                correctly_detect = 0
                object_height = 0
                object_width = 0
                percentage = 0 
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    if cur_object["name"] == true_class:
                        correctly_detect = 1
                        object_width  = cur_object['xmax'] - cur_object['xmin']
                        object_height = cur_object['ymax'] - cur_object['ymin']
                        object_size = object_width * object_height 
                        percentage = object_size / (image_height * image_width)
                    object_num += 1
                self.append_to_log([image_id, model, true_class, correctly_detect, object_num, 
                                    image_width, image_height, object_width, object_height, percentage], 'inference_log')
        self.append_to_log([image_id, response_time], 'request_log')
    def append_to_log(self, data, file_name):
        path = "../logs/{}.csv".format(file_name)
        if not os.path.exists(path):
            with open(path, "w") as f:
                writer = csv.writer(f)
                if file_name == "request_log":
                    writer.writerow(self.request_log_comlums)
                elif file_name == 'inference_log':
                    writer.writerow(self.inference_log_columns)
        with open(path, "a") as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(data)

if __name__ == '__main__':
    connector_config = load_config("./client_connector.json")
    client = Client(connector_config)
    i = 0 
    try:
        while i < 1000:
            client.send_message()
            i += 1
        #client.send_message()
    except KeyboardInterrupt:
        print('interrupted!')
# Multi-thrad
# for i in range(concurrent):
#     t = Thread(target=sender,args=[i])
#     t.start()
