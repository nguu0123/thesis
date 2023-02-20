from concurrent.futures import thread
from email.policy import default
import json, psutil, time, os, docker
from qoa4ml.reports import Qoa_Client
from threading import Thread

###################### DEFAULT METRIC ######################
default_docker_metric = {
    "dock_cpu_percentage":{
        "class": "Gauge",
        "description": "monitor system cpu percentage",
        "default": -1,
        "key": "percentage"
    },
    "docker_memory_used": {
        "class": "Gauge",
        "description": "monitor system memory used",
        "default": - 0,
        "key": "used"
    }
}


###################### COMMON USED FUNCTION ######################
def load_config(file_path:str)->dict:
    """
    file_path: file path to load config
    """
    with open(file_path, "r") as f:
        return json.load(f)

def to_json(file_path:str, conf:dict):
    """
    file_path: file path to save config
    """
    with open(file_path, "w") as f:
        json.dump(conf, f)
    
def get_sys_cpu():
    stats = psutil.cpu_stats()
    cpu_time = psutil.cpu_times()
    info = {}
    for key in stats._fields:
        info[key] = getattr(stats,key)
    for key in cpu_time._fields:
        info[key] = getattr(cpu_time,key)
    return info

def get_sys_mem():
    stats = psutil.virtual_memory()
    info = {}
    for key in stats._fields:
        info[key] = getattr(stats,key)
    return info

def get_sys_net():
    info = {}
    net = psutil.net_io_counters()
    for key in net._fields:
        info[key] = getattr(net,key)
    return info

def report_proc_cpu(process):
    report = {}
    cpu_time = process.cpu_times()
    contex = process.num_ctx_switches()
    for key in cpu_time._fields:
        report[key] = getattr(cpu_time,key)
    for key in contex._fields:
        report[key] = getattr(contex,key)
    report['num_thread'] = process.num_threads()

    return report
    
def get_proc_cpu(pid = None):
    if (pid == None):
        pid = os.getpid()
    process = psutil.Process(pid)
    child_list = process.children()
    info = {}
    info[pid] = report_proc_cpu(process)
    
    for child in child_list:
        info[child.pid + "c"] = report_proc_cpu(child)
    return info

def report_proc_mem(process):
    report = {}
    mem_info = process.memory_info()
    for key in mem_info._fields:
        report[key] = getattr(mem_info,key)
    return report
    
def get_proc_mem(pid = None):
    if (pid == None):
        pid = os.getpid()
    process = psutil.Process(pid)
    child_list = process.children()
    info = {}
    info[pid] = report_proc_mem(process)
    
    for child in child_list:
        info[child.pid + "c"] = report_proc_mem(child)
    return info

###################### SYSTEM REPORT ######################

sys_monitor_flag = False
proc_monitor_flag = False
doc_monitor_flag = False

def system_report(client:Qoa_Client, interval:int):
    report = {}
    while sys_monitor_flag:
        try:
            report["sys_cpu_stats"] = get_sys_mem()
        except Exception as e:
            print("Unable to report CPU stat: ", e)
        try:
            report["sys_mem_stats"] = get_sys_mem()
        except Exception as e:
            print("Unable to report memory stat: ", e)
        try:
            report["sys_net_stats"] = get_sys_net()
        except Exception as e:
            print("Unable to report network stat: ", e)
        try:
            client.report(report=report)
        except Exception as e:
            print("Unable to send system report: ", e)
        time.sleep(interval)


def sys_monitor(client:Qoa_Client, interval:int):
    sub_thread = Thread(target=system_report, args=(client, interval))
    sub_thread.start()


###################### PROCESS REPORT ######################   

def process_report(client:Qoa_Client, interval:int, pid:int = None):
    report = {}
    while proc_monitor_flag:
        try:
            report["proc_cpu_stats"] = get_proc_cpu()
        except Exception as e:
            print("Unable to report process CPU stat: ", e)
        try:
            report["proc_mem_stats"] = get_proc_mem()
        except Exception as e:
            print("Unable to report process memory stat: ", e)
        try:
            client.report(report=report)
        except Exception as e:
            print("Unable to send process report: ", e)
        time.sleep(interval)


def process_monitor(client:Qoa_Client, interval:int, pid:int = None):
    if (pid == None):
        pid = os.getpid()
    sub_thread = Thread(target=process_report, args=(client, interval, pid))
    sub_thread.start()


###################### DOCKER REPORT ######################

def get_cpu_stat(stats,key):
    if key == 'percentage':
        UsageDelta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']

        SystemDelta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']

        len_cpu = stats['cpu_stats']['online_cpus']

        percentage = (UsageDelta / SystemDelta) * len_cpu * 100
        return round(percentage, 2)
    else:
        return -1
def get_mem_stat(stats,key):
    if key == "used":
        return stats["memory_stats"]["usage"]
    else:
        return -1

        
def get_docker_stats(client):
    stats = {}
    try:
        for container in client.containers.list():
            stats[container.name] = {}
            stats[container.name]["cpu"] = {}
            stats[container.name]["mem"]= {}

            stat = container.stats(decode=None, stream = False) 
            stats[container.name]["cpu"]["percentage"] = get_cpu_stat(stat,"percentage")
            stats[container.name]["mem"]["used"] = get_mem_stat(stat,"used")
    except Exception as e:
        print("Unable to query docker stat: ", e)
    return stats

def docker_report(client:Qoa_Client, interval:int, metrics:dict = None, detail = False):
    try:
        client.add_metric(metrics)
    except Exception as e:
        print("Unable to add docker metric: ", e)
    metric_list = list(metrics.keys())
    doc_client = docker.from_env()
    
    while doc_monitor_flag:
        sum_cpu = 0
        sum_memory = 0
        try:
            stats = get_docker_stats(doc_client)
            for container_name in stats:
                sum_cpu += stats[container_name]["cpu"][metrics['dock_cpu_percentage']['key']]
                sum_memory += stats[container_name]["mem"][metrics['docker_memory_used']['key']]
                
            cpu_metric = client.get_metric('dock_cpu_percentage')
            cpu_metric.set(sum_cpu)
            mem_metric = client.get_metric('docker_memory_used')
            mem_metric.set(sum_memory)
        except Exception as e:
            print("Unable to report docker stat: ", e)

        try:
            if detail:
                client.report(report=stats)
            else:
                client.report(metrics=metric_list)
        except Exception as e:
            print("Unable to send system report: ", e)
        time.sleep(interval)

def docker_monitor(client:Qoa_Client, interval:int, metrics: dict = None, detail=False):
    if (metrics == None):
        metrics = default_docker_metric
    sub_thread = Thread(target=docker_report, args=(client, interval, metrics,detail))
    sub_thread.start()
