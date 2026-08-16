import os
import logging
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split


logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    with open("params.yaml" , "r") as f:
        params = yaml.safe_load(f)
        logging.info("Parameters loaded successfully")

except Exception as e:
    logging.info(f"Error occured {e}")

def data_ingestion(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        logging.info("Removed Items")

        return df

    except Exception as e:
        logging.info(f"Error occured {e}")


def save_data(df: pd.DataFrame , file_path):
    try:
        os.makedirs(os.path.dirname(file_path) , exist_ok= True)
        df.to_csv(file_path , index=False)
        logging.info(f"Data Saved Successfully to {file_path}")

    except Exception as e:
        logging.info(f"Error Occured {e}")


def main():
    try:
        file_path_csv = os.path.join("ai4i2020.csv")
        data = data_ingestion(file_path_csv)

        file_path = os.path.join("data" , "processed" , "cleaned_data.csv")
        save_data(df=data , file_path = file_path)

        logging.info("Data is Cleaned")

    except Exception as e:
        logging.info("Error Occured")



