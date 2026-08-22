from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import numpy as np
import pandas as pd
import pickle
import os
import logging

import uvicorn

from Machine_failure import Machine_failure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Contextual Predictive Maintenance")

# ----------------------------------------------------------------------
# Load artefacts once at startup. If either fails we crash immediately
# rather than serving an app that 500s on every request.
# ----------------------------------------------------------------------
MODEL_PATH = os.path.join("model", "Xgboost_model.pkl")
ENCODING_PATH = os.path.join("model", "encoding.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully: %s", type(model).__name__)

    with open(ENCODING_PATH, "rb") as f:
        encoding = pickle.load(f)
    logger.info("Encoder loaded successfully: %s", type(encoding).__name__)

    # Useful on startup: tells you what column names the encoder expects.
    expected = getattr(encoding, "feature_names_in_", None)
    if expected is not None:
        logger.info("Encoder expects columns: %s", list(expected))

except Exception as e:
    logger.exception("Failed to load artefacts: %s", e)
    raise

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    """Quick check that the process is up and the model is in memory."""
    return {"status": "ok", "model": type(model).__name__}


@app.post("/predict_api")
def predict_failure(data: Machine_failure):
    d = data.model_dump()

    try:
        # Column names matter: a scikit-learn transformer fitted on a named
        # DataFrame will warn or fail if it receives columns 0 and 1.
        val = pd.DataFrame(
            [[d["Product_ID"], d["Type"]]],
        )

        a = encoding.transform(val)
        if hasattr(a, "toarray"):          # sparse output from OneHotEncoder
            a = a.toarray()
        a = np.asarray(a).reshape(1, -1)

        features = np.array([[
            d["Air_temperature_K"],
            d["Process_temperature_K"],
            d["Rotational_speed_rpm"],
            d["Torque_Nm"],
            d["Tool_wear_min"],
            d["TWF"], d["HDF"], d["PWF"], d["OSF"], d["RNF"],
        ]], dtype=float)
        X = np.hstack([a, features])

        # .ravel()[0] works whether predict returns a scalar, (1,) or (1, 1).
        pred = int(np.asarray(model.predict(X)).ravel()[0])

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(np.asarray(model.predict_proba(X)).ravel()[-1])

    except Exception as e:
        logger.exception("Prediction failed for payload %s", d)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    logger.info(
        "Product_ID=%s Type=%s -> machine_failure=%s proba=%s",
        d["Product_ID"], d["Type"], pred,
        f"{proba:.4f}" if proba is not None else "n/a",
    )

    # Plain Python types only. A numpy array or np.int64 here is what caused
    # the "cannot convert dictionary update sequence element #0" error.
    return {"machine_failure": pred, "probability": proba}


if __name__ == "__main__":
    # Import string + reload=True so edits take effect without a manual
    # restart. Passing the app object directly disables reloading.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)