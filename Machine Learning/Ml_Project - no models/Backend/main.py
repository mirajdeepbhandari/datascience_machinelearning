from fastapi import FastAPI
import joblib

from schemas.request import TextRequest
from schemas.response import PredictionResponse
from utils.utility import preprocess_text

app = FastAPI()

# Load trained pipeline
label_encoder = joblib.load(
    r"C:\z_Learn\miraj_pandas\Ml_Project\Backend\models\label_encoder.pkl"
)
pipeline = joblib.load(
    r"C:\z_Learn\miraj_pandas\Ml_Project\Backend\models\knn_model.pkl"
)


@app.get("/")
def read_root():
    return {"health": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):

    # Preprocess input text
    processed_text = preprocess_text(request.text)

    # Predict category
    prediction = pipeline.predict([processed_text])
    print(f"Raw prediction: {prediction}")

    prediction = prediction[0]
    
    prediction = label_encoder.inverse_transform([prediction])
    print(f"Predicted category: {prediction}")

    return PredictionResponse(category=prediction[0])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
