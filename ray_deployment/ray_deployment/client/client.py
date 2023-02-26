import requests,time, argparse
from threading import Thread
import os, random,json, cv2, threading
import numpy as np
from qoa4ml.connector.amqp_connector import Amqp_Connector
from qoa4ml.collector.amqp_collector import Amqp_Collector
from qoa4ml.utils import load_config
parser = argparse.ArgumentParser(description="Data processing")
parser.add_argument('--th', help='number concurrent thread', default=1)
parser.add_argument('--sl', help='time sleep', default=-1.0)
args = parser.parse_args()
concurrent = int(args.th)
time_sleep = float(args.sl)
# url = 'http://192.168.1.148:5000/inference_aaltosea'
url = 'http://127.0.0.1:8000/'


def sender(num_thread):
    count = 0
    start_time = time.time()
    while (time.time() - start_time < 600):
        print("This is thread: ",num_thread, "Starting request: ", count)
        ran_file = random.choice(os.listdir("./image"))
        files = {'image': open("./image/"+ran_file, 'rb')}
        response = requests.post(url, files=files)
        prediction = response.json()
        print(prediction["prediction"]["yolo5"])
        print(prediction["prediction"]["yolo8"])
        image = np.asarray((prediction["image"]))

        # Comment this to use multi-thread
        cv2.imshow("lable",image.astype(np.uint8))
        cv2.waitKey(2000)
        time.sleep(10000)


        print("Thread - ",num_thread, " Response:" )
        count += 1
        if time_sleep == -1:
            time.sleep(1)
        else:
            time.sleep(time_sleep)


# 1 Thread

class Client():
    def __init__(self, connector_config: dict, collector_config: dict, log:bool = False):
        self.amqp_connector = Amqp_Connector(connector_config, log)
        self.amqp_collector = Amqp_Collector(collector_config, self)
        self.sub_thread = threading.Thread(target=self.amqp_collector.start)
        self.sub_thread.start()
    
    def send_message(self):
        print("sent")
        start_time = time.time()
        ran_class = random.choice(os.listdir("./images/"))
        ran_file = random.choice(os.listdir("./images/{}".format(ran_class)))
        files = open("./images/"+ran_class+'/'+ran_file, 'rb').read()
        files = bytearray(files)
        self.amqp_connector.send_data(files, 'client1')
    def message_processing(self, ch, method, props, body):
        data = json.loads(body)
        if data:
            output = data['prediction']['yolov5s'][0]['object_0'][0]
            print(output['name'], output['confidence'], '\n')
        else:
            print("cant detect")
connector_config = load_config('./client_connector.json')
collector_config = load_config('./client_collector.json')
client = Client(connector_config, collector_config)
for i in range(20):
    client.send_message()
# Multi-thrad
# for i in range(concurrent):
#     t = Thread(target=sender,args=[i])
#     t.start()

