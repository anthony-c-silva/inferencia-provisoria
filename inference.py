import onnxruntime as rt
import numpy as np
from config import MODEL_PATH
from models import BacteriasFeatures, PredictionResult

session = rt.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name

def predict(data: BacteriasFeatures) -> PredictionResult:
    """Recebe os dados já formatados e retorna a predição."""
    
    value = np.array([[float(data.TempoPontoInflexao)]], dtype=np.float32)
    
    predicted_coli_totais_raw = session.run([label_name], {input_name: value})[0]
    predict_e_coli = -0.0129 * value[0][0] + 10.492
    
    result = PredictionResult(
        predict_ecoli=float(predict_e_coli), 
        predict_colitotais=float(predicted_coli_totais_raw[0][0])
    )
    
    print(f"✅ Resultado da predição: {result}")
    return result