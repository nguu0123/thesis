import requests, time, argparse, csv
from threading import Thread
import os, random, json, cv2, threading
import numpy as np
from memory_profiler import profile
from qoa4ml.connector.amqp_connector import Amqp_Connector
from qoa4ml.collector.amqp_collector import Amqp_Collector
from qoa4ml.utils import load_config
import requests
from concurrent.futures import ThreadPoolExecutor
import time
random.seed(1234)
parser = argparse.ArgumentParser(description="Data processing")
parser.add_argument("--th", help="number concurrent thread", default=1)
parser.add_argument("--sl", help="time sleep", default=-1.0)
args = parser.parse_args()
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
    def __init__(self):
        # self.amqp_connector = Amqp_Connector(connector_config, log)
        self.correct_detect = 0
        self.request_sent = 0 
        self.request_log_comlums   = ["Image ID", "RequestId", "Model", "correctly_detect", "true_class"]
        self.inference_log_columns = ["Image ID", "Model", "Class", "Correctly detect", "Number of objects detected", 'Width', 
                                      'Height', 'Object width', 'Object height', 'Object to Image percentage']
    def send_message(self):
        ran_class = random.choice(os.listdir("./images_with_noise_04/"))
        ran_file = random.choice(os.listdir("./images_with_noise_04/{}".format(ran_class)))
        file = open("./images_with_noise_04/" + ran_class + "/" + ran_file, "rb").read()
        start_time = time.time()
        response = requests.post(url, files={"data": file, "file name": ran_file})
        response_time = time.time() - start_time
        prediction = response.json()
        correctly_detect = 0
        for model in prediction["prediction"].keys():
            if prediction["prediction"][model]:
                data = prediction["prediction"][model] 
                object_num = 0
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    if cur_object["name"] == ran_class: 
                        correctly_detect = 1
                    object_num += 1
        self.append_to_log([ran_file, response_time, correctly_detect, ran_class], "accuracy_without_noise_04" )
        #np_array = np.frombuffer(file, np.uint8)
        #im = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        #height, width, _ = im.shape
        #self.monitor(time.time() - start_time, prediction["prediction"], ran_file, ran_class, height, width)
    def send_message_with_models(self):
        ran_class = random.choice(os.listdir("./images_in_bright/"))
        ran_file = random.choice(os.listdir("./images_in_bright/{}".format(ran_class)))
        file = open("./images_in_bright/" + ran_class + "/" + ran_file, "rb").read()
        start_time = time.time()
        response = requests.post(url, files={"data": file, "file name": ran_file})
        response_time = time.time() - start_time
        prediction = response.json()
        for model in prediction["prediction"].keys():
            correctly_detect = 0
            if prediction["prediction"][model]:
                data = prediction["prediction"][model] 
                object_num = 0
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    if cur_object["name"] == ran_class: 
                        correctly_detect = 1
                    object_num += 1
            self.append_to_log([ran_file, prediction["requestId"], model, correctly_detect, ran_class], "mean_accuracy_with_prov_v5n_v8n" )
    def send_message_without_brightness(self):
        ran_class = random.choice(os.listdir("./images/"))
        ran_file = random.choice(os.listdir("./images/{}".format(ran_class)))
        file = open("./images/" + ran_class + "/" + ran_file, "rb").read()
        start_time = time.time()
        response = requests.post(url, files={"data": file, "file name": ran_file})
        response_time = time.time() - start_time
        prediction = response.json()
        for model in prediction["prediction"].keys():
            correctly_detect = 0
            if prediction["prediction"][model]:
                data = prediction["prediction"][model] 
                object_num = 0
                for detected_object in data:
                    cur_object = detected_object["object_{}".format(object_num)]
                    if isinstance(cur_object, list):
                        cur_object = cur_object[0]
                    if cur_object["name"] == ran_class: 
                        correctly_detect = 1
                    object_num += 1
            self.append_to_log([ran_file, prediction["requestId"], model, correctly_detect, ran_class], "mean_accuracy_with_prov_v5n_v8n" )
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
                else: 
                    writer.writerow(self.request_log_comlums)
        with open(path, "a") as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(data)

num_requests = 1000
# Define a function to send requests
def send_request():
    for i in range(1):
        try:
            ran_class = random.choice(os.listdir("./images/"))
            ran_file = random.choice(os.listdir("./images/{}".format(ran_class)))
            file = open("./images/" + ran_class + "/" + ran_file, "rb").read()
            response = requests.post(url, files={"data": file, "file name": ran_file})
        except Exception as e:
            print(f'Error sending request {i+1} to {url}: {e}')

def send_requests(rps, duration):

    def send_request():
        ran_class = random.choice(os.listdir("./images_with_noise/"))
        ran_file = random.choice(os.listdir("./images_with_noise/{}".format(ran_class)))
        file = open("./images/" + ran_class + "/" + ran_file, "rb").read()
        response = requests.post(url, files={"data": file, "file name": ran_file})

    start_time = time.time()
    interval = 1 / rps
    total_requests = rps * duration

    # Use ThreadPoolExecutor to limit concurrent threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(send_request) for _ in range(total_requests)]

    end_time = time.time()
    total_time = end_time - start_time

    print(f'Total Requests: {total_requests}')
    print(f'Total Time (seconds): {total_time}')
    print(f'Requests per Second: {total_requests / total_time}')

if __name__ == '__main__':
## Use ThreadPoolExecutor to send requests concurrently
    #send_requests(30,20) 
    #send_request()
    # Wait for all tasks to complete
    i = 0
    client = Client()
    try:
        while i < 300:
             client.send_message_without_brightness()
             i += 1
        i = 0
        while i < 300:
             client.send_message_with_models()
             i += 1
        #client.send_message()
    except KeyboardInterrupt:
        print('interrupted!')
# Multi-thrad
# for i in range(concurrent):
#     t = Thread(target=sender,args=[i])
#     t.start()
