import time
import uuid
import json
import functools
import asyncpg
import os
import psutil
import psycopg
import threading
import tracemalloc
from datetime import datetime
from qoa4ml.utils import load_config, get_proc_cpu, get_proc_cpu, thread
from torch.nn.functional import linear


config = (
        "user=nguu0123 password=nguu0123456 host=172.17.0.3 port=5432 dbname=nguu0123"
    )
class Data:
    def __init__(self, data, id=None, name=None, requestId=None):
        self.data = data
        self.id = str(uuid.uuid4()) if id is None else id
        self.name = "test"
        self.requestId = requestId

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return Data(returnData, self.id)


class Prediction:
    def __init__(self, prediction, id=None, requestId=None):
        self.data = prediction
        self.id = str(uuid.uuid4()) if id is None else id
        self.name = "test"
        self.requestId = requestId

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return Prediction(returnData, self.id)


class DataQualityReport:
    def __init__(self, report, id=None, requestId=None):
        self.data = report
        self.id = str(uuid.uuid4()) if id is None else id
        self.requestId = requestId

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return DataQualityReport(returnData, self.id)


class PredictionQuality:
    def __init__(self, metrics, id=None, requestId=None):
        self.data = metrics
        self.id = str(uuid.uuid4()) if id is None else id
        self.requestId = requestId

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return DataQualityReport(returnData, self.id)


def capturePredictActivity(
    activityId,
    startTime,
    endTime,
    data: Data,
    prediction: Prediction,
    predictionCpu,
    predictionMem,
    modelId,
):
    global config
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    postgres_insert_query = (
        """ INSERT INTO predict (id, start_time, end_time, requestId) VALUES (%s,%s,%s,%s)"""
    )
    record_to_insert = (activityId, startTime, endTime, data.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = """ INSERT INTO prediction (id, name, requestId) VALUES (%s,%s,%s)"""
    record_to_insert = (prediction.id, prediction.name, data.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
    )
    record_to_insert = (activityId, prediction.id)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
    record_to_insert = (activityId, data.id)
    cursor.execute(postgres_insert_query, record_to_insert)

    predictionQuality = {"cpu": predictionCpu, "mem": predictionMem}
    if "QoA" in prediction.data:
        predictionQuality["QoA"] = prediction.data["QoA"]
    predictionQualityId = str(uuid.uuid4())
    postgres_insert_query = (
        """ INSERT INTO predictionquality(id, value, requestId) VALUES (%s,%s,%s)"""
    )
    record_to_insert = (predictionQualityId, json.dumps(predictionQuality), data.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
    )
    record_to_insert = (activityId, predictionQualityId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO wasassociatedwith(activityid, agentid) VALUES (%s,%s)"""
    )
    record_to_insert = (activityId, modelId)
    cursor.execute(postgres_insert_query, record_to_insert)


def captureAssessDataQualityActivity(
     activityId, data: Data, dataQualityReport: DataQualityReport
):
    global config
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    postgres_insert_query = (
        """ INSERT INTO dataqualityreport(id, value, requestId) VALUES (%s,%s,%s)"""
    )
    record_to_insert = (dataQualityReport.id, json.dumps(dataQualityReport.data), data.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO assessdataquality(id, name, requestId) VALUES (%s,%s,%s)"""
    )
    record_to_insert = (activityId, "assess data quality", data.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
    )
    record_to_insert = (activityId, dataQualityReport.id)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
    record_to_insert = (activityId, data.id)
    cursor.execute(postgres_insert_query, record_to_insert)


def capturePreprocessingActivity(
    activityId, inputData: Data, outputData: Data, funcName
):
    global config
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    postgres_insert_query = """ INSERT INTO preprocess(id, name, requestId) VALUES (%s,%s,%s)"""
    record_to_insert = (activityId, funcName, inputData.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = """ INSERT INTO data(id, name, requestId) VALUES (%s,%s,%s)"""
    record_to_insert = (outputData.id, funcName + " output", inputData.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
    )
    record_to_insert = (activityId, outputData.id)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
    record_to_insert = (activityId, inputData.id)
    cursor.execute(postgres_insert_query, record_to_insert)


def captureEnsembleActivity(
    activityId, inputs: list[Prediction], output: Prediction, funcName
):
    global config
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    postgres_insert_query = """ INSERT INTO ensemblefunction(id, name, requestId) VALUES (%s,%s,%s)"""
    record_to_insert = (activityId, funcName, output.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = """ INSERT INTO prediction(id, name, requestId) VALUES (%s,%s,%s)"""
    record_to_insert = (output.id, funcName + "result", output.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)

    postgres_insert_query = (
        """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
    )
    record_to_insert = (activityId, output.id)
    cursor.execute(postgres_insert_query, record_to_insert)

    for input in inputs:
        postgres_insert_query = (
            """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, input.id)
        cursor.execute(postgres_insert_query, record_to_insert)


def getModelId(modelName, param):
    global config
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    postgreSQL_select_Query = (
        "select * from model where name = %s and parameter ->> 'param' = %s"
    )
    cursor.execute(postgreSQL_select_Query, (modelName, param))
    mobile_records = cursor.fetchall()
    if cursor.rowcount == 0:
        modelId = str(uuid.uuid4())
        postgres_insert_query = (
            """ INSERT INTO model(id, name, parameter) VALUES (%s,%s,%s)"""
        )
        record_to_insert = (modelId, modelName, json.dumps({"param": param}))
        cursor.execute(postgres_insert_query, record_to_insert)
        return modelId
    return mobile_records[0][0]


def report_proc_cpu(process):
    report = {}
    cpu_time = process.cpu_times()
    contex = process.num_ctx_switches()
    for key in cpu_time._fields:
        report[key] = getattr(cpu_time, key)
    for key in contex._fields:
        report[key] = getattr(contex, key)
    report["num_thread"] = process.num_threads()

    return report


def get_proc_cpu(pid=None):
    if pid == None:
        pid = os.getpid()
    process = psutil.Process(pid)
    return report_proc_cpu(process)


def report_proc_mem(process):
    report = {}
    mem_info = process.memory_info()
    for key in mem_info._fields:
        report[key] = getattr(mem_info, key)
    return report


def get_proc_mem(pid=None):
    if pid == None:
        pid = os.getpid()
    process = psutil.Process(pid)
    return report_proc_mem(process)


def capture(activityType):
    def wrapper(func):
        @functools.wraps(func)
        def doFunc(*args, **kwargs):
            startTime = datetime.now()
            beforeCpu = None
            predictionMem = None
            if activityType == "predict":
                beforeCpu = get_proc_cpu()
                tracemalloc.start()
            returnVal = func(*args, **kwargs)
            if activityType == "predict":
                predictionMem = tracemalloc.get_traced_memory()
                predictionCpu = {"before": beforeCpu, "after": get_proc_cpu()}
                tracemalloc.stop()
            funcName = func.__name__
            endTime = datetime.now()
            activityId = str(uuid.uuid4())
            if activityType == "predict":
                data = kwargs["data"]
                returnVal = Prediction(returnVal,None, data.requestId)
                async_thread = threading.Thread(
                    target=capturePredictActivity,
                    args= (
                        activityId,
                        startTime,
                        endTime,
                        data,
                        returnVal,
                        predictionCpu,
                        predictionMem,
                        args[0].id,
                    )
                )
                async_thread.start()
            elif activityType == "assessDataQuality":
                data = kwargs["data"]
                returnVal = DataQualityReport(returnVal, data.requestId)
                async_thread = threading.Thread(
                    target=captureAssessDataQualityActivity,
                    args=(activityId, data, returnVal
                    )
                )
                async_thread.start()
            elif activityType == "preprocessing":
                inputData = kwargs["data"]
                returnVal = Data(returnVal,None, None, inputData.requestId)
                async_thread = threading.Thread(
                    target=capturePreprocessingActivity,
                    args=(activityId, inputData, returnVal, funcName
                    )
                )
                async_thread.start()
            elif activityType == "ensemble":
                inputPredictions = kwargs["predictions"]
                returnVal = Prediction(returnVal, None, inputPredictions[0].requestId)
                async_thread = threading.Thread(
                    target=captureEnsembleActivity,
                    args=
                        (activityId, inputPredictions, returnVal, funcName
                    )
                )
                async_thread.start()
            return returnVal

        return doFunc

    return wrapper


def captureModel(func):
    @functools.wraps(func)
    def wrapper_init_model(*args, **kwargs):
        model = args[0]
        param = kwargs["param"]
        model.id = getModelId(model.__class__.__name__, param)
        return func(*args, **kwargs)

    return wrapper_init_model


def captureInputData(data):
    global config
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    returnData = Data(data, None, None, str(uuid.uuid4()))
    postgres_insert_query = """ INSERT INTO data(id, name, requestId) VALUES (%s,%s,%s)"""
    record_to_insert = (returnData.id, "input data", returnData.requestId)
    cursor.execute(postgres_insert_query, record_to_insert)
    return returnData
