from pathlib import Path
import sqlite3
import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API for the Heart Attack Risk Prediction project",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "heart_attack_prediction_india.csv"
DB_PATH = BASE_DIR / "predictions.db"

# Patient_ID is deliberately excluded: it is an identifier, not a health feature.
TARGET = "Heart_Attack_Risk"
K_NEIGHBORS = 3

df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    raise ValueError(f"Dataset must contain target column {TARGET!r}")

X = df.drop(columns=[TARGET, "Patient_ID"], errors="ignore")
y = df[TARGET]

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ],
    remainder="drop",
)

# Train from the dataset with the same preprocessing used for live user inputs.
# K=3 is used instead of K=30, which heavily favored class 0 in the original model.
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=K_NEIGHBORS)),
    ]
)
model.fit(X, y)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_data TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


init_db()


class PatientData(BaseModel):
    Age: float
    Gender: str
    State_Name: str
    Annual_Income: float
    Health_Insurance: str
    Diabetes: str
    Hypertension: str
    Obesity: str
    Smoking: str
    Alcohol_Consumption: str
    Physical_Activity: str
    Diet_Score: float
    Cholesterol_Level: float
    Triglyceride_Level: float
    LDL_Level: float
    HDL_Level: float
    Systolic_BP: float
    Diastolic_BP: float
    Air_Pollution_Exposure: str
    Family_History: str
    Stress_Level: str
    Healthcare_Access: str
    Emergency_Response_Time: float
    Heart_Attack_History: str


@app.get("/")
def home():
    return {"message": "Heart Attack Risk Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "database": str(DB_PATH),
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "KNeighborsClassifier",
        "best_k": K_NEIGHBORS,
        "features": len(X.columns),
        "excluded_feature": "Patient_ID",
    }


@app.post("/predict")
@app.post("/predict")
def predict(patient: PatientData):
    try:
        input_data = patient.model_dump()

        # Convert Yes/No values to numbers
        yes_no_mapping = {
            "Yes": 1,
            "No": 0
        }

        categorical_binary_columns = [
            "Diabetes",
            "Hypertension",
            "Obesity",
            "Smoking",
            "Alcohol_Consumption",
            "Physical_Activity",
            "Air_Pollution_Exposure",
            "Family_History",
            "Healthcare_Access",
            "Heart_Attack_History",
            "Health_Insurance"
        ]

        for col in categorical_binary_columns:
            value = input_data[col]

            if isinstance(value, str):
               value = value.strip().title()

               if value in yes_no_mapping:
                  input_data[col] = yes_no_mapping[value]
               elif value in ("0", "1"):
                  input_data[col] = int(value)
               else:
                  raise ValueError(
                    f"Unexpected value {value!r} for column {col}"
                )

        # Convert stress level to numeric
        stress_mapping = {
            "Low": 3,
            "Medium": 6,
            "High": 9
        }

        value = input_data["Stress_Level"]

        if isinstance(value, str):
           value = value.strip().title()

           if value in stress_mapping:
             input_data["Stress_Level"] = stress_mapping[value]
           elif value.isdigit() and 1 <= int(value) <= 10:
             input_data["Stress_Level"] = int(value)
           else:
             raise ValueError(f"Unexpected Stress_Level: {value!r}")
        # Create the input DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=X.columns)

        # Apply preprocessing and predict
        input_processed = preprocessor.transform(input_df)
        result = int(
            model.named_steps["knn"].predict(input_processed)[0]
        )

        prediction = str(result)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions (input_data, prediction)
                VALUES (?, ?)
                """,
                (patient.model_dump_json(), prediction),
            )

        return {
            "prediction": prediction,
            "model": f"K-Nearest Neighbors (k={K_NEIGHBORS})",
            "message": "Prediction generated successfully",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/history")
def prediction_history(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, input_data, prediction, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []
    for row in rows:
        record = dict(row)
        try:
            record["patient_data"] = json.loads(record.pop("input_data"))
        except Exception:
            record["patient_data"] = {}
        history.append(record)

    return {"history": history}
