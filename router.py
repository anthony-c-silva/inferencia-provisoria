from fastapi import APIRouter, Depends, Body
from typing import List

from models import PredictionResult, ProcessarEntrada, BacteriasFeatures
from security import validate_request
from processing import process_and_predict
from inference import predict as predict_direct

router = APIRouter()

@router.post("/processar", 
             response_model=PredictionResult, 
             summary="Processa dados do experimento e retorna a predição de bactérias")
async def endpoint_processar(dados_entrada: ProcessarEntrada):
    """
    Recebe os IDs de um experimento, busca e processa os dados,
    e retorna a predição final de E. coli e Coliformes Totais.
    Este é o endpoint principal para integração com o CRM.
    """
    # A lógica complexa foi movida para o service `processing.py`
    resultado = process_and_predict(dados_entrada)
    return resultado

@router.post("/predict", 
             response_model=PredictionResult,
             dependencies=[Depends(validate_request)],
             summary="Executa a predição direta a partir das features calculadas")
def endpoint_predict(features: BacteriasFeatures):
    """
    Recebe as 7 features já calculadas e retorna a predição.
    Este endpoint requer um token de autenticação no header.
    """
    # Este endpoint chama a predição diretamente
    return predict_direct(features)