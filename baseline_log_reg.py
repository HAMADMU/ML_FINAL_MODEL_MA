import numpy as np              #Handles arrays and math
import pandas as pd             #reads data and manipulates table

from sklearn.model_selection import train_test_split    #splits data into training and test sets

from sklearn.pipeline import Pipeline                   #chains steps together in order
from sklearn.impute import SimpleImputer                #fills missing values
from sklearn.preprocessing import StandardScaler        #scales data for the model
from sklearn.linear_model import LogisticRegression     #baseline classifier

from sklearn.metrics import (
    confusion_matrix,               #confusion matrix
    classification_report,          #precision/recall
    average_precision_score,        #PR-AUC
    precision_recall_curve,         #threshold tuning     *FLAG ALL*
)

DATA = "IEEE68bus_ML_ready_risky.csv.gz"        #compressed dataset file name
RANDOM_STATE = 42                               #makes split reproducible *FLAG*

def main():
    df = pd.read_csv(DATA, compression="gzip")      #reads compressed file
    print("loaded:", df.shape, flush=True)          #show the number of columns/rows

    y = df["risky"].astype(int)             #output (what I want the model to predict)
    x = df.drop(columns=["risky"])          #operating point features

    print("Class balance:", y.value_counts().to_dict(), flush=True)         
    print("Rate of risk:", float(y.mean()), flush=True)                 #How imbalanced the data is

    #----------------------------------------------
    #80% train, 20% test.
    #----------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y     #stratify makes a split so that the proportion... 
                                                                       #...of values in the sample produced is the same as...
                                                                       #...the proportion of values provided by parameters 
                                                                       #...maintain 1% risk
    )
    
    #----------------------------------------------
    #Baseline model pipeline
    #----------------------------------------------
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),                                      #
        ("scalar", StandardScaler(with_mean=False)),                                        #scales features to be on similar numeric ranges, with_mean=False: avoids issue with dense metrics
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced" ))         #predicts probability of risk, class_weight="balanced": gives risk cases more importance during training due to rareness 
    ]                                                                                       #...otherwise model might ignore risk cases
    )

    model.fit(x_train, y_train)             #Model learns patterns from training data

    