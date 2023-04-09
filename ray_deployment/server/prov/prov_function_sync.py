import time
import uuid
import json
import functools
import asyncpg
import os
import psutil
import psycopg
from datetime import datetime
from qoa4ml.utils import load_config, get_proc_cpu, get_proc_cpu
from torch.nn.functional import linear


class Data:
    def __init__(self, data, id=None, name=None):
        self.data = data
        self.id = str(uuid.uuid4()) if id is None else id
        self.name = "test"

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return Data(returnData, self.id)


class Prediction:
    def __init__(self, prediction, id=None):
        self.data = prediction
        self.id = str(uuid.uuid4()) if id is None else id
        self.name = "test"

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return Prediction(returnData, self.id)


class DataQualityReport:
    def __init__(self, report, id=None):
        self.data = report
        self.id = str(uuid.uuid4()) if id is None else id

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return DataQualityReport(returnData, self.id)


class PredictionQuality:
    def __init__(self, metrics, id=None):
        self.data = metrics
        self.id = str(uuid.uuid4()) if id is None else id

    def getAs(self, *args):
        if type(self.data) == dict:
            returnData = {}
            for key in args:
                if key in self.data:
                    returnData = returnData | self.data[key]
            return DataQualityReport(returnData, self.id)


class LineageManager:
    def __init__(self) -> None:
        config = "user=nguu0123 password=nguu0123456 host=172.17.0.2 port=5432 dbname=nguu0123"
        self.connection = psycopg.connect(config, autocommit=True)
        self.cursor = self.connection.cursor()

    def capturePredictActivity(
        self,
        activityId,
        startTime,
        endTime,
        data: Data,
        prediction: Prediction,
        predictionCpu,
        predictionMem,
        modelId,
    ):
        postgres_insert_query = (
            """ INSERT INTO predict (id, start_time, end_time) VALUES (%s,%s,%s)"""
        )
        record_to_insert = (activityId, startTime, endTime)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = """ INSERT INTO prediction (id, name) VALUES (%s,%s)"""
        record_to_insert = (prediction.id, prediction.name)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, prediction.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, data.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        predictionQuality = {"cpu": predictionCpu, "mem": predictionMem}
        if "QoA" in prediction.data:
            predictionQuality["QoA"] = prediction.data["QoA"]
        predictionQualityId = str(uuid.uuid4())
        postgres_insert_query = (
            """ INSERT INTO predictionquality(id, value) VALUES (%s,%s)"""
        )
        record_to_insert = (predictionQualityId, json.dumps(predictionQuality))
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, predictionQualityId)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO wasassociatedwith(activityid, agentid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, modelId)
        self.cursor.execute(postgres_insert_query, record_to_insert)

    def captureAssessDataQualityActivity(
        self, activityId, data: Data, dataQualityReport: DataQualityReport
    ):
        postgres_insert_query = (
            """ INSERT INTO dataqualityreport(id, value) VALUES (%s,%s)"""
        )
        record_to_insert = (dataQualityReport.id, json.dumps(dataQualityReport.data))
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO assessdataquality(id, name) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, "assess data quality")
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, dataQualityReport.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, data.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

    def capturePreprocessingActivity(
        self, activityId, inputData: Data, outputData: Data, funcName
    ):
        postgres_insert_query = """ INSERT INTO preprocess(id, name) VALUES (%s,%s)"""
        record_to_insert = (activityId, funcName)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = """ INSERT INTO data(id, name) VALUES (%s,%s)"""
        record_to_insert = (outputData.id, funcName + "output")
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, outputData.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, inputData.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

    def captureEnsembleActivity(
        self, activityId, inputs: list[Prediction], output: Prediction, funcName
    ):
        postgres_insert_query = (
            """ INSERT INTO ensemblefunction(id, name) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, funcName)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = """ INSERT INTO prediction(id, name) VALUES (%s,%s)"""
        record_to_insert = (output.id, funcName + "result")
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, output.id)
        self.cursor.execute(postgres_insert_query, record_to_insert)

        for input in inputs:
            postgres_insert_query = (
                """ INSERT INTO used(activityid, entityid) VALUES (%s,%s)"""
            )
            record_to_insert = (activityId, input.id)
            self.cursor.execute(postgres_insert_query, record_to_insert)

    def getModelId(self, modelName, param):
        postgreSQL_select_Query = (
            "select * from model where name = %s and parameter ->> 'param' = %s"
        )
        self.cursor.execute(postgreSQL_select_Query, (modelName, param))
        mobile_records = self.cursor.fetchall()
        if self.cursor.rowcount == 0:
            modelId = str(uuid.uuid4())
            postgres_insert_query = (
                """ INSERT INTO model(id, name, parameter) VALUES (%s,%s,%s)"""
            )
            record_to_insert = (modelId, modelName, json.dumps({"param": param}))
            self.cursor.execute(postgres_insert_query, record_to_insert)
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
            beforeMem = None
            if activityType == "predict":
                beforeCpu = get_proc_cpu()
                beforeMem = get_proc_mem()
            returnVal = func(*args, **kwargs)
            funcName = func.__name__
            endTime = datetime.now()
            activityId = str(uuid.uuid4())
            lineageManager = LineageManager()
            if activityType == "predict":
                predictionCpu = {"before": beforeCpu, "after": get_proc_cpu()}
                predictionMem = {"before": beforeMem, "after": get_proc_mem()}
                data = kwargs["data"]
                returnVal = Prediction(returnVal)
                lineageManager.capturePredictActivity(
                    activityId,
                    startTime,
                    endTime,
                    data,
                    returnVal,
                    predictionCpu,
                    predictionMem,
                    args[0].id,
                )
            elif activityType == "assessDataQuality":
                data = kwargs["data"]
                returnVal = DataQualityReport(returnVal)
                lineageManager.captureAssessDataQualityActivity(
                    activityId, data, returnVal
                )
            elif activityType == "preprocessing":
                inputData = kwargs["data"]
                returnVal = Data(returnVal)
                lineageManager.capturePreprocessingActivity(
                    activityId, inputData, returnVal, funcName
                )
            elif activityType == "ensemble":
                inputPredictions = kwargs["predictions"]
                returnVal = Prediction(returnVal)
                lineageManager.captureEnsembleActivity(
                    activityId, inputPredictions, returnVal, funcName
                )
            return returnVal

        return doFunc

    return wrapper


def captureModel(func):
    @functools.wraps(func)
    def wrapper_init_model(*args, **kwargs):
        model = args[0]
        lineageManager = LineageManager()
        param = kwargs["param"]
        model.id = lineageManager.getModelId(model.__class__.__name__, param)
        return func(*args, **kwargs)

    return wrapper_init_model


def captureInputData(data):
    returnData = Data(data)
    lineageManager = LineageManager()
    postgres_insert_query = """ INSERT INTO data(id, name) VALUES (%s,%s)"""
    record_to_insert = (returnData.id, "input data")
    lineageManager.cursor.execute(postgres_insert_query, record_to_insert)
    return returnData
