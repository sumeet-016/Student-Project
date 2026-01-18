import sys 
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is used for Tranformation of Data
        '''
        try:
            numeric_feature = ["writing score", "reading score"]
            categoric_feature = [
                "gender", "race/ethnicity",
                "parental level of education",
                "lunch", "test preparation course"
            ]

            num_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoding", OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("numerical_pipeline", num_pipe, numeric_feature),
                ("categorical_pipe", cat_pipe, categoric_feature)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def inititate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column = "math score"
            numeric_feature = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Appling preprocessing object on training dataframe and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved preprocessing object")
 
            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)