from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
from function import categorical
from sklearn.metrics import classification_report
from sklearn.utils import resample
from typing import List
from fastapi.responses import JSONResponse
import openai 

openai.api_key = ""

app = FastAPI(
    title="API Machine Learning",
    description="API pour l'entraînement et la prédiction avec un modèle de machine learning",
    version="1.0.0"
)

class TrainRequest(BaseModel):
    data: list
    target: str

class PredictRequest(BaseModel):
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

        # Retirer la colonne CustomerID
        df = df.drop(columns=['CustomerID'])
        
        # Séparer les caractéristiques (X) et la cible (y)
        X = df.drop(columns=[request.target])
        y = df[request.target]

        # Combiner X et y pour le rééchantillonnage
        df_combined = pd.concat([X, y], axis=1)
        
        # Séparer les classes majoritaires et minoritaires
        df_majority = df_combined[df_combined[request.target] == 0]
        df_minority = df_combined[df_combined[request.target] == 1]

        # Suréchantillonnage de la classe minoritaire
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,     # Échantillonnage avec remplacement
                                         n_samples=len(df_majority),    # Pour égaler la classe majoritaire
                                         random_state=123) # Reproductibilité

        # Combiner les données suréchantillonnées
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        
        # Séparer les caractéristiques (X) et la cible (y) après rééchantillonnage
        X = df_upsampled.drop(columns=[request.target])
        y = df_upsampled[request.target]

        # Appliquer get_dummies pour transformer les colonnes catégorielles en variables indicatrices
        X = pd.get_dummies(X, columns=['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit', 'DiningOccasion', 'MealType'])

        # Entraîner le modèle
        model = LogisticRegression()
        model.fit(X, y)
        
        # Évaluer le modèle
        y_pred = model.predict(X)
        report = classification_report(y, y_pred)
        print(report)
        
        # Sauvegarder les colonnes du modèle
        model_columns = list(X.columns)
        joblib.dump(model_columns, "model/model_columns.pkl")
        
        # Sauvegarder le modèle entraîné
        if not os.path.exists("model"):
            os.makedirs("model")
        joblib.dump(model, "model/model.pkl")
        
        return {"message": "Model trained successfully", "classification_report": report}
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
        
        # Appliquer get_dummies pour transformer les colonnes catégorielles en variables indicatrices
        input_data = pd.get_dummies(input_data, columns=['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit', 'DiningOccasion', 'MealType'])
        
        # Aligner les colonnes de input_data avec celles utilisées lors de l'entraînement du modèle
        model_columns = joblib.load("model/model_columns.pkl")
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Faire la prédiction
        prediction = model.predict(input_data)
        
        # Convertir le type de la prédiction en un type natif Python
        prediction = prediction.tolist()
        
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/model", tags=["Model Info"])
async def get_model_response(text: str = Query(..., title="Text to send to the model")):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
        )
        return JSONResponse(content={"response": response.choices[0].message['content']})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from OpenAI: {e}")

@app.exception_handler(Exception)
async def validation_exception_handler(request, err):
    return JSONResponse(
        status_code=400,
        content={"message": f"An error occurred: {err}"}
    )
    

