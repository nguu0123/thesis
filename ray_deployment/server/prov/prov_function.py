import time
import uuid
import functools
import psycopg2
from datetime import datetime
from qoa4ml.utils import load_config


class Data:
    def __init__(self, data) -> None:
        self.data = data
        self.id = str(uuid.uuid4())
        self.name = "test"


class Prediction:
    def __init__(self, prediction):
        self.data = prediction
        self.id = str(uuid.uuid4())
        self.name = "test"


class DataQualityReport:
    def __init__(self, report) -> None:
        self.data = report
        self.id = str(uuid.uuid4())


class PredictionQuality:
    def __init__(self, metrics):
        self.data = metrics
        self.id = str(uuid.uuid4())


class LineageManager:
    def __init__(self, connectorConfig) -> None:
        self.connection = psycopg2.connect(
            user=connectorConfig["user"],
            password=connectorConfig["password"],
            host=connectorConfig["host"],
            port=connectorConfig["port"],
            database=connectorConfig["database"],
            autocommit=True,
        )
        self.cursor = self.connection.cursor()

    def capturePredictActivity(
        self, activityId, startTime, endTime, data: Data, prediction: Prediction
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

    def captureAssessDataQualityActivity(
        self, activityId, data: Data, dataQualityReport: DataQualityReport
    ):
        postgres_insert_query = """ INSERT INTO dataqualityreport(value) VALUES (%s)"""
        record_to_insert = dataQualityReport.data
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = (
            """ INSERT INTO assessdataquality(id, name) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, "test")
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
        self, activityId, inputData: Data, outputData: Data
    ):
        postgres_insert_query = """ INSERT INTO preprocess(id, name) VALUES (%s,%s)"""
        record_to_insert = (activityId, "test")
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = """ INSERT INTO data(id, name) VALUES (%s,%s)"""
        record_to_insert = (outputData.id, "preprocessing output")
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
        self, activityId, inputs: list[Prediction], output: Prediction
    ):
        postgres_insert_query = (
            """ INSERT INTO ensemblefunction(id, name) VALUES (%s,%s)"""
        )
        record_to_insert = (activityId, "ensemble test")
        self.cursor.execute(postgres_insert_query, record_to_insert)

        postgres_insert_query = """ INSERT INTO prediction(id, name) VALUES (%s,%s)"""
        record_to_insert = (output.id, "ensemble result")
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


def capture(activityType):
    def wrapper(func):
        @functools.wraps(func)
        def doFunc(*args, **kwargs):
            config = {
                "user": "nguu0123",
                "password": "nguu0123456",
                "host": "172.17.0.2",
                "port": "5432",
                "database": "nguu0123",
            }
            lineageManager = LineageManager(config)
            startTime = datetime.now()
            returnVal = func(*args, **kwargs)
            endTime = datetime.now()
            activityId = str(uuid.uuid4())
            if activityType == "predict":
                data = Data(kwargs["data"])
                returnVal = Prediction(returnVal)
                lineageManager.capturePredictActivity(
                    activityId, startTime, endTime, data, returnVal
                )
            elif activityType == "assessDataQuality":
                pass
            elif activityType == "preprocessing":
                inputData = kwargs["data"]
                outputData = Data(returnVal)
                lineageManager.capturePreprocessingActivity(
                    activityId, inputData, outputData
                )
            elif activityType == "ensemble":
                pass
            return returnVal

        return doFunc

    return wrapper


def captureInputData(data):
    config = {
        "user": "nguu0123",
        "password": "nguu0123456",
        "host": "172.17.0.2",
        "port": "5432",
        "database": "nguu0123",
    }
    lineageManager = LineageManager(config)
    returnData = Data(data)
    postgres_insert_query = """ INSERT INTO data(id, name) VALUES (%s,%s)"""
    record_to_insert = (returnData.id, "input data")
    lineageManager.cursor.execute(postgres_insert_query, record_to_insert)
    return returnData
