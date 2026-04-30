import numpy as np              #Handles arrays and math
import pandas as pd             #reads data and manipulates table
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split    #splits data into training and test sets


from sklearn.impute import SimpleImputer                #fills missing values
from sklearn.preprocessing import StandardScaler        #scales data for the model
from sklearn.linear_model import LogisticRegression     #baseline classifier

from sklearn.metrics import (
    confusion_matrix,               #confusion matrix
    classification_report,          #precision/recall
    average_precision_score,        #PR-AUC (Precision: When the model says risky, how often is it correct?, recall: Out of all truly risky cases, how many did it catch?)
    precision_recall_curve,         #threshold tuning     *FLAG ALL*
    f1_score,
    recall_score,
)

from xgboost import XGBClassifier 

DATA = "IEEE68bus_ML_ready_risky.csv.gz"        #compressed dataset file name
RANDOM_STATE = 42                               #makes split reproducible *FLAG*


#------------------------------------
# Confusion matrix plot function
#------------------------------------
def plot_confusion(cm, title, filename):                
    fig, ax = plt.subplots()                        #creates matplotlib figure with axis
    im = ax.imshow(cm)                              #displays as colored image

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["not_risky", "risky"])
    ax.set_yticklabels(["not_risky", "risky"])

    for (i, j), v in np.ndenumerate(cm):                    #writes count into cell
        ax.text(j, i, str(v), ha="center", va="center")     #vertical and horizontal alignment 

    fig.colorbar(im, ax=ax)                                 #color scale bar
    plt.savefig(filename, dpi=300, bbox_inches="tight")     #saves figure
    plt.show()                                              #displays

#------------------------------------
# Precision-recall plot function
#------------------------------------

def plot_pr_curve(y_true, proba, filename, thr_default=0.5, thr_tuned=None):        
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
    plt.grid(True)

    # Mark chosen thresholds on the PR curve
    def mark(th, label):
        if len(thresholds) == 0:
            return
        idx = np.argmin(np.abs(thresholds - th))
        # recall/precision arrays are one element longer than thresholds
        plt.scatter(recall[idx + 1], precision[idx + 1], s=60)
        plt.text(recall[idx + 1], precision[idx + 1], label)

    mark(thr_default, f"th={thr_default}")
    if thr_tuned is not None:
        mark(thr_tuned, f"th={thr_tuned:.3f}")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


#MAIN CODE
def main():
    df = pd.read_csv(DATA, compression="gzip")      #reads compressed file
    print("loaded:", df.shape, flush=True)          #show the number of columns/rows

    y = df["risky"].astype(int)             #output (what I want the model to predict)
    x = df.drop(columns=["risky"])          #operating point features

    print("Class balance:", y.value_counts().to_dict(), flush=True)         
    print("Rate of risk:", float(y.mean()), flush=True)                 #How imbalanced the data is

    #----------------------------------------------
    #80% train, 30% test.
    #----------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y     #stratify makes a split so that the proportion... 
                                                                       #...of values in the sample produced is the same as...
                                                                       #...the proportion of values provided by parameters 
                                                                       #...maintain 1% risk
    

    )
    
    #----------------------------------------------
    #Baseline model pipeline
    #----------------------------------------------
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),                                      
        ("smote", SMOTE(random_state=42)),   # balance minority class
        ("scaler", StandardScaler(with_mean=False)),                                        #scales features to be on similar numeric ranges, with_mean=False: avoids issue with dense metrics
        ("classifier", LogisticRegression(max_iter=3000,))         #predicts probability of risk, class_weight="balanced": gives risk cases more importance during training due to rareness 
    ]                                                                                       #...otherwise model might ignore risk cases   *FLAG*
    )

    model.fit(x_train, y_train)             #Model learns patterns from training data

    proba = model.predict_proba(x_test)[:,1]        #predict_proba: returns PROBABILITY for both classes: class 0 then class 1 (not-risky, risky)
    pred = (proba >= 0.5).astype(int)               #[:,1]: selects predicted risky points
                                                    #risky if probability >=0.5


    #----------------------------------------------
    #Evaluation
    #----------------------------------------------
    print("PR-AUC:", average_precision_score(y_test, proba))            #higher(>0.01(risky rate)) PR-AUC means model is learning
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))        #Helps to find false negative
    print(classification_report(y_test, pred, digits=4))                #compares truth vs prediction and prints metrics in 4 decimal places

    #--------------------------------------------
    #Threshold tuning
    #--------------------------------------------

    target_recall = 0.90
    precision, recall, thresholds = precision_recall_curve(y_test, proba)       #computes precision and recall for many thresholds

    best_thr = None
    best_f1 = -1.0

    for th in thresholds: 
        pred_temp = (proba >= th).astype(int)
        r = recall_score(y_test, pred_temp, zero_division=0)
        f1 = f1_score(y_test, pred_temp, zero_division=0)

        if r >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_thr = th

    if best_thr is None:
        print(f"\nNo threshold found with recall >= {target_recall:.2f}")
        best_thr = 0.5

    pred_thr = (proba >= best_thr).astype(int)                                       #replaces default threshold with tuned threshold

    #------------------------------------------------
    #Printing threshold labels
    #------------------------------------------------
    print(f"\n--- Threshold tuned evaluation (target recall {target_recall:.2f}) ---")
    print("Chosen threshold:", best_thr)
    print("Confusion matrix:\n", confusion_matrix(y_test, pred_thr))
    print(classification_report(y_test, pred_thr, digits=4))

    #----------------------------------------
    #VISUALIZATION 
    #----------------------------------------
    cm_default = confusion_matrix(y_test, pred)             #prediction using default threshold
    cm_tuned = confusion_matrix(y_test, pred_thr)           #prediction using tuned threshold

    plot_confusion(cm_default, "Confusion Matrix (threshold = 0.5)", "fig_confusion_05.png")
    plot_confusion(cm_tuned, f"Confusion Matrix (threshold = {best_thr:.3f})", "fig_confusion_tuned.png")

    plot_pr_curve(y_test, proba, "fig_pr_curve.png", thr_default=0.5, thr_tuned=best_thr)

    #Predicted Probability Distribution
    plt.figure()
    plt.hist(proba[y_test == 0], bins=40, alpha=0.7, label="not_risky")
    plt.hist(proba[y_test == 1], bins=40, alpha=0.7, label="risky")
    plt.yscale("log")
    plt.xlabel("Predicted P(risky)")
    plt.ylabel("Count (log scale)")
    plt.title("Predicted Probability Distribution (log y-scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig("fig_proba_hist_logy.png", dpi=300, bbox_inches="tight")
    plt.show()


    #Feature Importance
    result = permutation_importance(
    model, x_test, y_test,
    n_repeats=5, random_state=42, scoring="average_precision"
    )

    importances = pd.Series(result.importances_mean, index=x_test.columns).sort_values(ascending=False).head(20)

    plt.figure()
    plt.barh(importances.index[::-1], importances.values[::-1])
    plt.xlabel("Permutation importance (Δ PR-AUC)")
    plt.title("Top 20 Feature Importances (Baseline)")
    plt.tight_layout()
    plt.savefig("fig_feat_importance_top20.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()