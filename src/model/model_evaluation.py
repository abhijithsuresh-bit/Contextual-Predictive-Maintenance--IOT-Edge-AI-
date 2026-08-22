import os
import pandas as pd
import logging
import pickle
import json
from sklearn.metrics import classification_report , accuracy_score

logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def model_evaluation(X_test_path , y_test_path , model_path , evalation_json_path):
    try:
        with open(model_path , "rb") as f:
            model = pickle.load(f)
        logging.info("model loaded successfully ")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        pred = model.predict(X_test)

        classification_repo = classification_report(y_true= y_test , y_pred= pred , output_dict=True)

        os.makedirs(os.path.dirname(evalation_json_path) , exist_ok=True)

        with open(evalation_json_path , "w")as f:
            json.dump(classification_repo , f , indent= 4)

    except Exception as e:
        print(f"Error Occured {e}")

def main():
    try:
        X_test_path = os.path.join("data" ,"preprocessed" , "X_test.csv")
        y_test_path = os.path.join("data" ,"preprocessed" , "y_test.csv")

        model_path = os.path.join("model" , "Xgboost_model.pkl")

        evaluation_report_path = os.path.join("evaluation" , "evaluation_report.json")

        model_evaluation(X_test_path= X_test_path ,  y_test_path= y_test_path , model_path= model_path , evalation_json_path= evaluation_report_path)

    except Exception as e:
        print(f"Error Occured {e}")


if __name__ == "__main__":
    main()


        
