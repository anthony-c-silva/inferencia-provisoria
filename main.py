from fastapi import FastAPI
import uvicorn
from router import router

app = FastAPI(
    title="BioAI Lab - API Unificada",
    description="API para processamento de dados de experimentos e predição de bactérias.",
    version="1.0.0"
)

# Inclui os endpoints do arquivo router.py
app.include_router(router)

@app.get("/", summary="Endpoint raiz da API")
def read_root():
    return {"status": "API Unificada da BioAI Lab está online"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)