import pandas as pd 
import logging
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import os
import pickle


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    with open("params.yaml" , "r") as f:
        params = yaml.safe_load(f)
        logging.info("Params loaded successfully")

except Exception as e:
    logging.info(f"Error Occured {e}")


def model_building(X_train_file_path , y_train_file_path , file_path):
    try:
        X_train = pd.read_csv(X_train_file_path)
        y_train = pd.read_csv(y_train_file_path)

        sm = SMOTE(random_state=42)
        X_train_res , y_train_res = sm.fit_resample(X_train , y_train)

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train_res, y_train_res.squeeze())

        os.makedirs(os.path.dirname(file_path) , exist_ok= True)

        with open(file_path , "wb") as f:
            pickle.dump(model , f)

            logging.info("Model successfully saved")

    except Exception as e:
        logging.info(f"Error orrcured {e}")

def main():
    try:
        X_train_file_path = os.path.join("data" , "preprocessed" , "X_train.csv")
        y_train_file_path = os.path.join("data" , "preprocessed" , "y_train.csv")

        model_file_path = os.path.join("model" , "Xgboost_model.pkl")

        model_building(X_train_file_path , y_train_file_path , model_file_path)

        logging.info("Savved Sucessfully")

    except Exception as e:
        logging.info(f"Error orrcured {e}")


if __name__ == "__main__":
    main()









