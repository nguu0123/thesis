import time
import uuid
import json
import functools
import asyncpg
import os
import psutil
import psycopg
import asyncio
import asyncpg
from datetime import datetime
from qoa4ml.utils import load_config, get_proc_cpu, get_proc_cpu
from torch.nn.functional import linear
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio

nest_asyncio.apply()
config = "postgresql://nguu0123:nguu0123456@172.17.0.3:5432/nguu0123"


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


async def capturePredictActivity(
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
    connection = await asyncpg.connect(config)
    async with connection.transaction():
        await connection.execute(
            """ INSERT INTO predict (id, start_time, end_time) VALUES ($1,$2,$3)""",
            activityId,
            startTime,
            endTime,
        )

        await connection.execute(
            """ INSERT INTO prediction (id, name) VALUES ($1,$2)""",
            prediction.id,
            prediction.name,
        )

        await connection.execute(
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            prediction.id,
        )

        await connection.execute(
            """ INSERT INTO used(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            data.id,
        )

        predictionQuality = {"cpu": predictionCpu, "mem": predictionMem}
        if "QoA" in prediction.data:
            predictionQuality["QoA"] = prediction.data["QoA"]
        predictionQualityId = str(uuid.uuid4())

        await connection.execute(
            """ INSERT INTO predictionquality(id, value) VALUES ($1,$2)""",
            predictionQualityId,
            json.dumps(predictionQuality),
        )

        await connection.execute(
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            predictionQualityId,
        )

        await connection.execute(
            """ INSERT INTO wasassociatedwith(activityid, agentid) VALUES ($1,$2)""",
            activityId,
            modelId,
        )
    await connection.close()


async def captureAssessDataQualityActivity(
    activityId, data: Data, dataQualityReport: DataQualityReport
):
    global config
    connection = await asyncpg.connect(config)
    async with connection.transaction():
        await connection.execute(
            """ INSERT INTO dataqualityreport(id, value) VALUES ($1,$2)""",
            dataQualityReport.id,
            json.dumps(dataQualityReport.data),
        )

        await connection.execute(
            """ INSERT INTO assessdataquality(id, name) VALUES ($1,$2)""",
            activityId,
            "assess data quality",
        )

        await connection.execute(
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            dataQualityReport.id,
        )

        await connection.execute(
            """ INSERT INTO used(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            data.id,
        )
    await connection.close()


async def capturePreprocessingActivity(
    activityId, inputData: Data, outputData: Data, funcName
):
    global config
    connection = await asyncpg.connect(config)
    async with connection.transaction():
        await connection.execute(
            """ INSERT INTO preprocess(id, name) VALUES ($1,$2)""", activityId, funcName
        )

        await connection.execute(
            """ INSERT INTO data(id, name) VALUES ($1,$2)""",
            outputData.id,
            funcName + " output",
        )

        await connection.execute(
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            outputData.id,
        )

        await connection.execute(
            """ INSERT INTO used(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            inputData.id,
        )
    await connection.close()


async def captureEnsembleActivity(
    activityId, inputs: list[Prediction], output: Prediction, funcName
):
    global config
    connection = await asyncpg.connect(config)
    async with connection.transaction():
        await connection.execute(
            """ INSERT INTO ensemblefunction(id, name) VALUES ($1,$2)""",
            activityId,
            funcName,
        )

        await connection.execute(
            """ INSERT INTO prediction(id, name) VALUES ($1,$2)""",
            output.id,
            funcName + "result",
        )

        await connection.execute(
            """ INSERT INTO wasgeneratedby(activityid, entityid) VALUES ($1,$2)""",
            activityId,
            output.id,
        )

        for input in inputs:
            await connection.execute(
                """ INSERT INTO used(activityid, entityid) VALUES ($1,$2)""",
                activityId,
                input.id,
            )
    await connection.close()


def getModelId(modelName, param):
    config = (
        "user=nguu0123 password=nguu0123456 host=172.17.0.3 port=5432 dbname=nguu0123"
    )
    connection = psycopg.connect(config, autocommit=True)
    cursor = connection.cursor()
    postgreSQL_select_Query = (
        "select * from model where name = %s and parameter ->> 'param' = %s"
    )
    cursor.execute(postgreSQL_select_Query, (modelName, param))
    modelRecords = cursor.fetchall()
    if cursor.rowcount == 0:
        modelId = str(uuid.uuid4())
        postgres_insert_query = (
            """ INSERT INTO model(id, name, parameter) VALUES (%s,%s,%s)"""
        )
        record_to_insert = (modelId, modelName, json.dumps({"param": param}))
        cursor.execute(postgres_insert_query, record_to_insert)
        return modelId
    return modelRecords[0][0]


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
            if activityType == "predict":
                predictionCpu = {"before": beforeCpu, "after": get_proc_cpu()}
                predictionMem = {"before": beforeMem, "after": get_proc_mem()}
                data = kwargs["data"]
                returnVal = Prediction(returnVal)
                asyncio.new_event_loop().run_in_executor(None,
                    capturePredictActivity(
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
            elif activityType == "assessDataQuality":
                data = kwargs["data"]
                returnVal = DataQualityReport(returnVal)
                asyncio.new_event_loop().run_in_executor(None,
                    captureAssessDataQualityActivity(activityId, data, returnVal)
                )
            elif activityType == "preprocessing":
                inputData = kwargs["data"]
                returnVal = Data(returnVal)
                asyncio.new_event_loop().run_in_executor(None,
                    capturePreprocessingActivity(
                        activityId, inputData, returnVal, funcName
                    )
                )
            elif activityType == "ensemble":
                inputPredictions = kwargs["predictions"]
                returnVal = Prediction(returnVal)
                asyncio.new_event_loop().run_in_executor(None,
                    captureEnsembleActivity(
                        activityId, inputPredictions, returnVal, funcName
                    )
                )
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


async def captureInputData(data):
    returnData = Data(data)
    global config
    conn = await asyncpg.connect(config)
    await conn.execute(
        """ INSERT INTO data(id, name) VALUES ($1,$2)""", returnData.id, "input data"
    )
    await conn.close()
    return returnData
