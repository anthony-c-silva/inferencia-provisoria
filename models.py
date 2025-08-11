from pydantic import BaseModel
from typing import List

# Modelo de entrada para o endpoint /processar (da API 1)
class ProcessarEntrada(BaseModel):
    experimentId: str
    sensorId: str
    mathModelId: str
    sensorChannel: str
    sensorSubChannel: str
    analysisId: str

# Modelo de features para a predição (da API 2)
class BacteriasFeatures(BaseModel):
    Amplitude: float
    TempoPontoInflexao: float
    PontoInflexao: float
    TempoPicoPrimeiraDerivada: float
    PicoPrimeiraDerivada: float
    TempoPicoSegundaDerivada: float
    PicoSegundaDerivada: float

# Modelo do resultado da predição (da API 2)
class PredictionResult(BaseModel):
    predict_ecoli: float
    predict_colitotais: float