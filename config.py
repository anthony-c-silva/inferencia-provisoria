import os

# Configuração do MongoDB (da API 1)
MONGO_URI = os.getenv(
    "SPRING_DATA_MONGODB_URI",
    "mongodb://admin:admin@54.80.66.121:27017/bioailab-crm?authSource=admin"
)
MONGO_DB = os.getenv("SPRING_DATA_MONGODB_DATABASE", "bioailab-crm")

# Configuração de Segurança e Modelo (da API 2)
TOKEN = os.getenv("TOKEN", "1337")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("model", "RANSACregression_richards_turbidimetria_G.onnx"))