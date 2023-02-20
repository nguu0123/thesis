from typing import List
from .connector.amqp_connector import Amqp_Connector
from .connector.mqtt_connector import Mqtt_Connector
from .connector.prom_connector import Prom_Connector
from .probes import Gauge, Counter, Summary, Histogram
import json, uuid
import threading
from threading import Thread
import time
from datetime import datetime

class Qoa_Client(object):
    # Init QoA Client
    def __init__(self, client_conf: dict, connector_conf: dict):
        '''
        Client configuration contains the information about the client and its configuration in form of dictionary
        Example: 
        { 
            "client_id": "aaltosea1",
            "component_id": "data_processing"
        }
        The `connector_conf` is the dictionary containing multiple connector configuration (amqp, mqtt, kafka)
        Example: 
        {
            "amqp_connector":{
                "class": "amqp",
                "conf":{
                    "end_point": "localhost",
                    "exchange_name": "qoa4ml",
                    "exchange_type": "topic",
                    "out_routing_key": "qoa.report.ml"
                }
            }
        }
        '''
        self.start_time = time.time()
        self.config = client_conf
        self.metrics = {}
        self.connector = {}
        # Init all connectors in for loop 
        for key in connector_conf:
            self.connector[key] = self.init_connector(connector_conf[key])
        
        # Set default connector for sending monitoring data if not specify
        self.default_connector = list(self.connector.keys())[0]
        # self.add_metric(metric_conf)
        self.lock = threading.Lock()
        

    def init_connector(self, configuration: dict):
        if configuration["class"] == "amqp":
            return Amqp_Connector(configuration["conf"])
        if configuration["class"] == "mqtt":
            return Mqtt_Connector(configuration["conf"])
        # if configuration["class"] == "kafka":
        #     return Kafka_Connector(configuration["conf"])
        
    # def add_metric(self, metric_conf: dict, category=None):
    #     # Add multiple metrics 
    #     if (category == None):
    #         # category = "default"
    #         pass
    #     else:
    #         if category in self.metrics:
    #             pass
    #         else:
    #             self.metrics[category] = {}
    #         for key in metric_conf:
    #             if key in self.metrics[category]:
    #                 pass
    #             else:
    #                 self.metrics[category][key] = self.init_metric(key, metric_conf[key])
    def add_metric(self, metric_conf: dict):
        # Add multiple metrics 
        for key in metric_conf:
            self.metrics[key] = self.init_metric(key, metric_conf[key])

    def init_metric(self, name, configuration: dict):
        # init individual metrics
        if configuration["class"] == "Gauge":
            return Gauge(name, configuration["description"], configuration["default"])
        elif configuration["class"] == "Counter":
            return Counter(name, configuration["description"], configuration["default"])
        elif configuration["class"] == "Summary":
            return Summary(name, configuration["description"], configuration["default"])
        elif configuration["class"] == "Histogram":
            return Histogram(name, configuration["description"], configuration["default"])

    def get(self):
        # TO DO:
        return self.config

    # def get_metric(self, key=None, category=None):
    #     # TO DO:
    #     if (category == None) & (key == None):
    #         metrics = {}
    #         for category in self.metrics:
    #             metrics.update(self.metrics[category])
    #         return metrics
    #     if (key == None) & (category != None):
    #         return self.metrics[category]
    #     if (isinstance(key, list)) & (category!=None):
    #         return dict((k, self.metrics[category][k]) for k in key)
    #     if (key!=None) & (category!=None): 
    #         return self.metrics[category][key]

    def get_metric(self, key=None):
        # TO DO:
        if key == None:
            return self.metrics
        elif isinstance(key, list):
            return dict((k, self.metrics[k]) for k in key)
        else: 
            return self.metrics[key]

    def set(self, key, value):
        # TO DO:
        try:
            self.config[key] = value
        except Exception as e:
            print("{} not found - {}".format(key,e))
    
    
    # def generate_report(self, metric:list=None, category=None):
    #     report = {}
    #     if (category == None):
    #         for category_key in self.metrics:
    #             report[category_key] = {}
    #             list_metric = list(self.metrics[category_key].keys())
    #             for key in list_metric:
    #                 report[category_key][key] = self.metrics[category_key][key].get_val()
    #     else:
    #         report[category] = {}
    #         if (metric == None):
    #             metric = list(self.metrics[category].keys())
    #         for key in metric:
    #             report[category][key] = self.metrics[category][key].get_val()
    #     return report
    def generate_report(self, metric:list=None):
        report = {}
        report["metric"] = {}
        if metric == None:
            metric = list(self.metrics.keys())
        for key in metric:
            report["metric"][key] = self.metrics[key].get_val()
        return report
    

    # def asyn_report(self, metrics:list=None, report:dict = None, connectors:list=None, category=None):
    #     client_inf = self.config.copy()
    #     client_inf['timestamp'] = str(datetime.fromtimestamp(time.time()))
    #     client_inf['runtime'] = time.time() - self.start_time
        
    #     if (report == None):
    #         report = self.generate_report(metrics, category)
    #     report.update(client_inf)
    #     body_mess = json.dumps(report)
    #     self.lock.acquire()
    #     if (connectors == None):
    #         self.connector[self.default_connector].send_data(body_mess,str(uuid.uuid4()))
    #     else:
    #         for connector in connectors:
    #             print(connector)
    #     self.lock.release()

    # def report(self, metrics:list=None, report: dict = None, connectors:list=None, category=None):
    #     sub_thread = Thread(target=self.asyn_report, args=(metrics,report, connectors, category))
    #     sub_thread.start()

    def asyn_report(self, metrics:list=None, report:dict = None, connectors:list=None):
        client_inf = self.config.copy()
        client_inf['timestamp'] = str(datetime.fromtimestamp(time.time()))
        client_inf['runtime'] = time.time() - self.start_time

        if (report == None):
            report = self.generate_report(metrics)
        report.update(client_inf)
        body_mess = json.dumps(report)
        self.lock.acquire()
        if connectors == None:
            self.connector[self.default_connector].send_data(body_mess,str(uuid.uuid4()))
        else:
            for connector in connectors:
                print(connector)
        self.lock.release()

    def report(self, metrics:list=None, report: dict = None, connectors:list=None):
        sub_thread = Thread(target=self.asyn_report, args=(metrics, report, connectors))
        sub_thread.start()


    def __str__(self):
        return str(self.config) + '\n' + str(self.connector)