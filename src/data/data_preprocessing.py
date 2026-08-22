import os
import logging
import yaml
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder


logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    with open("params.yaml" , "r") as f:
        params = yaml.safe_load(f)
        logging.info("Parameters loaded succesfully")

except Exception as e:
    logging.info(f"Error occured {e}")

def data_preprocessing(file_path):
    try:
        df = pd.read_csv(file_path)

        df.drop(columns=['UDI'] , axis=1 , inplace= True)

        X = df.drop(columns=["Machine failure"])
        y = df["Machine failure"]

        logging.info("Dat Preprocessing Successfully")

        return X , y



    except Exception as e:
        logging.info(f"Error occured {e}")


def split_data(X , y , test_size ):

    try:
        X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=test_size , random_state=42)

    except Exception as e:
        logging.info(f"Error occured {e}")

    return X_train , X_test , y_train , y_test


def converting_data_to_numeric(X_train , X_test , y_train , y_test , encoding_path):
    try:
        enc_auto = TargetEncoder(smooth="auto")
        X_train[["Product ID" , "Type"]] = enc_auto.fit_transform(X_train[["Product ID" , "Type"]] , y_train)
        X_test[["Product ID" , "Type"]] = enc_auto.transform(X_test[["Product ID" , "Type"]])

        os.makedirs(os.path.dirname(encoding_path) , exist_ok= True)
        with open(encoding_path , 'wb')as f:
            pickle.dump(enc_auto , f)


        logging.info("Data Converted successfully")

        return X_train , X_test
    
    except Exception as e:

        logging.info(f"Error occured {e}")


def save_data(df:pd.DataFrame , file_path):

    try:
        os.makedirs(os.path.dirname(file_path) , exist_ok= True)

        df.to_csv(file_path , index= False)

        logging.info("Data Stored Succesfully")

    except Exception as e:

        logging.info(f"Error occured {e}")


def main():
    try:

        data_path = os.path.join("data" , "processed" , "cleaned_data.csv")

        test_size = params["data_preprocessing"]["test_size"]

        X , y = data_preprocessing(file_path= data_path)

        X_train , X_test , y_train , y_test = split_data(X , y , test_size= test_size)

        enc_path = os.path.join("model" , "encoding.pkl")

        X_train , X_test = converting_data_to_numeric(X_train , X_test , y_train , y_test , encoding_path= enc_path)

        file_path_X_train = os.path.join("data" , "preprocessed" , "X_train.csv")
        file_path_X_test = os.path.join("data" , "preprocessed" , "X_test.csv")
        file_path_y_train = os.path.join("data" , "preprocessed" , "y_train.csv")
        file_path_y_test = os.path.join("data" , "preprocessed" , "y_test.csv")

        save_data(X_train , file_path_X_train)
        save_data(X_test , file_path_X_test)
        save_data(y_train , file_path_y_train)
        save_data(y_test , file_path_y_test)


    except Exception as e:
        logging.info(f"Error occured {e}")


if __name__ == "__main__":
    main()









