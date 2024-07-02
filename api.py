from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
from function import categorical
app = FastAPI(
    title="API Machine Learning",
    description="API pour l'entraînement et la prédiction avec un modèle de machine learning",
    version="1.0.0"
)

class TrainRequest(BaseModel):
    data: list
    target: str

class PredictRequest(BaseModel):
    CustomerID: int
    Age: int
    Gender: str
    Income: float
    VisitFrequency: str
    AverageSpend: float
    PreferredCuisine: str
    TimeOfVisit: str
    GroupSize: int
    DiningOccasion: str
    MealType: str
    OnlineReservation: int
    DeliveryOrder: int
    LoyaltyProgramMember: int
    WaitTime: float
    ServiceRating: int
    FoodRating: int
    AmbianceRating: int

@app.post("/training")
def train_model(request: TrainRequest):
    try:
        # Convertir les données en DataFrame
        df = pd.DataFrame(request.data)

        # Vérification de l'existence de la colonne cible
        if request.target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target}' not found in data")

        # Séparer les caractéristiques (X) et la cible (y)
        X = df.drop(columns=[request.target])
        y = df[request.target]
        
        # Identifier les colonnes catégorielles
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Appliquer la transformation catégorielle à chaque colonne catégorielle
        for col in categorical_features:
            X = categorical(X, col)
        
        # Entraîner le modèle
        model = LogisticRegression()
        model.fit(X, y)
        
        # Sauvegarder le modèle entraîné
        if not os.path.exists("model"):
            os.makedirs("model")
        joblib.dump(model, "model/model.pkl")
        
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        if not os.path.exists("model/model.pkl"):
            raise HTTPException(status_code=404, detail="Model not found")

        # Charger le modèle
        model = joblib.load("model/model.pkl")
        
        # Convertir la requête en DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Appliquer la transformation catégorielle aux colonnes de la requête
        categorical_features = input_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            input_data = categorical(input_data, col)
        
        # Faire la prédiction
        prediction = model.predict(input_data)
        
        # Convertir le type de la prédiction en un type natif Python
        prediction = prediction.tolist()
        
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/model")
def get_model_info():
    import requests

    url = "https://api-inference.huggingface.co/models/bert-base-uncased"
    headers = {"Authorization": "Bearer "}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()



