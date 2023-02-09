import requests,time, argparse
from threading import Thread
import os, random,json, cv2
import numpy as np


parser = argparse.ArgumentParser(description="Data processing")
parser.add_argument('--th', help='number concurrent thread', default=1)
parser.add_argument('--sl', help='time sleep', default=-1.0)
args = parser.parse_args()
concurrent = int(args.th)
time_sleep = float(args.sl)
# url = 'http://192.168.1.148:5000/inference_aaltosea'
url = 'http://0.0.0.0:5000/'


def sender(num_thread):
    count = 0
    start_time = time.time()
    while (time.time() - start_time < 600):
        print("This is thread: ",num_thread, "Starting request: ", count)
        ran_file = random.choice(os.listdir("./image"))
        files = {'image': open("./image/"+ran_file, 'rb'), 'user_data':open("./conf/client.json", 'rb')}
        response = requests.post(url, files=files)
        prediction = response.json()
        print(prediction["prediction"])
        image = np.asarray((prediction["image"]))

        # Comment this to use multi-thread
        cv2.imshow("lable",image.astype(np.uint8))
        cv2.waitKey(2000)


        print("Thread - ",num_thread, " Response:" )
        count += 1
        if time_sleep == -1:
            time.sleep(1)
        else:
            time.sleep(time_sleep)


# 1 Thread
sender(1)

# Multi-thread
# for i in range(concurrent):
#     t = Thread(target=sender,args=[i])
#     t.start()

