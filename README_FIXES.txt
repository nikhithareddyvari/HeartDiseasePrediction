# Heart Disease Prediction (corrected)

## Run the backend
Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Backend docs: http://127.0.0.1:8000/docs

## Run the frontend
Open `frontend/index.html` using VS Code Live Server (port 5500).

## What changed
- Retrains KNN from the supplied CSV when the backend starts.
- Excludes `Patient_ID` because it is an identifier and was not included in the form.
- Uses the same fitted preprocessing pipeline for training and prediction.
- Uses K=3 instead of K=30; K=30 predicted almost everything as class 0.
- Saves form values and the last result in browser localStorage so they can be restored after a reload.
- The form also prevents native submit/reload.
