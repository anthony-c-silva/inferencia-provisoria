from pymongo import MongoClient
from bson import ObjectId
from config import MONGO_URI, MONGO_DB # Importa do novo config.py

client = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]

def get_sensor(id: str):
    return db.sensors.find_one({"_id": ObjectId(id)})

def get_math_model(id: str):
    return db.growth_models.find_one({"_id": ObjectId(id)})

def get_experiment(id: str):
    return db.experiments.find_one({"_id": id})

def get_experiment_data(id: str, limit: int = 10_000):
    return list(
        db.data_analise
          .find({"experiment_id": id})
          .sort("timestamp", 1)
          .limit(limit)
    )