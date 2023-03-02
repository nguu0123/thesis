import requests, time, argparse
from threading import Thread
import os, random, json, cv2, threading
import numpy as np
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
    def __init__(
        self, connector_config: dict, collector_config: dict, log: bool = False
    ):
        self.amqp_connector = Amqp_Connector(connector_config, log)
        self.amqp_collector = Amqp_Collector(collector_config, self)
        self.sub_thread = threading.Thread(target=self.amqp_collector.start)
        self.sub_thread.start()
        self.correct_detect = 0
        self.request_sent = 0
        self.average_response_time = 0
        self.response_time = 0
        self.errors = 0

    def send_message(self):
        ran_class = random.choice(os.listdir("./images/"))
        ran_file = random.choice(os.listdir("./images/{}".format(ran_class)))
        files = open("./images/" + ran_class + "/" + ran_file, "rb").read()
        files = bytearray(files)
        request = {
            'file': files.hex(),
            'true_label': ran_class,
            'request_time': time.time()
        }
        self.amqp_connector.send_data(json.dumps(request), "client1")
        self.image_size = 640 * 480
        self.average_object_area_percentage = 0

    def message_processing(self, ch, method, props, body):
        self.request_sent += 1
        data = json.loads(body)
        image = np.asarray((data["image"]))

        # Comment this to use multi-thread
        cv2.imshow("lable", image.astype(np.uint8))
        cv2.waitKey(2000)
        object_area = 0
        try:
            aggregated_prediction = data['prediction']['aggregated'][0]
            for cur_object in aggregated_prediction.keys():
                print(aggregated_prediction[cur_object])
                object_area = (aggregated_prediction['xmax'] - aggregated_prediction['xmin']) * (aggregated_prediction['ymax'] - aggregated_prediction['ymin'])
                self.average_object_area_percentage += object_area / self.image_size
                print(aggregated_prediction['name'], aggregated_prediction['confidence'])
                if data['true_label'] == aggregated_prediction['name']:
                    print('correct detection')
                    self.correct_detect += 1
                else:
                    print('incorrect detection' )
        except:
            self.errors += 1 
        self.response_time = time.time() - data['request_time']
        self.average_response_time += self.response_time
        print("Response time {}".format(self.response_time))
        print("Avereage response time {}".format(self.average_response_time / self.request_sent))
        print('Accuracy {}'.format(self.correct_detect / self.request_sent))
        print('Avereage percentage of object in the image {}'.format(self.average_object_area_percentage / self.request_sent))
        print('Number of errors {}\n'.format(self.errors))
        time.sleep(10)
        self.send_message()
        


connector_config = load_config("./client_connector.json")
collector_config = load_config("./client_collector.json")
client = Client(connector_config, collector_config)
client.send_message()
# Multi-thrad
# for i in range(concurrent):
#     t = Thread(target=sender,args=[i])
#     t.start()
