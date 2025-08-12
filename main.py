# inferencia-provisoria/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 1. Importe aqui
import uvicorn
from router import router

app = FastAPI(
    title="BioAI Lab - API Unificada",
    description="API para processamento de dados de experimentos e predição de bactérias.",
    version="1.0.0"
)

# 2. Adicione o middleware de CORS aqui
origins = [
    "http://localhost",
    "http://localhost:3000",
    # Você pode adicionar outras origens permitidas aqui
    # Ex: "https://seu-dominio-de-producao.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

# Inclui os endpoints do arquivo router.py
app.include_router(router)

@app.get("/", summary="Endpoint raiz da API")
def read_root():
    return {"status": "API Unificada da BioAI Lab está online"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)