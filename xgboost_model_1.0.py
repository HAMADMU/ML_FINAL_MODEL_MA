import numpy as np                                                                     #math
import pandas as pd                                                                    #tables
import matplotlib.pyplot as plt                                                        #plots
 
from imblearn.pipeline import Pipeline                                                  #chains steps together 
from imblearn.over_sampling import SMOTE                                                #SMOTE creates synthetic risky points
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate   #used for the 5-fold cross-validation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold                                 #removes all features whose variance doesn't meet the threshold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve, 
    f1_score,
    recall_score,
)
 
from xgboost import XGBClassifier                                                       #Gradient boosting model
 
#-------------------------------------------------------------
#CONFIGURATION
#-------------------------------------------------------------
DATA          = "IEEE68bus_ML_ready_risky.csv.gz"                                       #same dataset as baseline script
RANDOM_STATE  = 42                                                                      #a seed number that makes all random operations reproducible
TOP_N_FEATS   = 100                                                                     #number of features to keep after selection
TARGET_RECALL = 0.90                                                                    #minimum recall we want after threshold tuning
 
 
#-------------------------------------------------------------
#confusion matrix plot
#-------------------------------------------------------------
def plot_confusion(cm, title, filename):
    fig, ax = plt.subplots()                                                            #creates matplotlib figure with axis
    im = ax.imshow(cm)                                                                  #displays as colored image
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["not_risky", "risky"])
    ax.set_yticklabels(["not_risky", "risky"])
    for (i, j), v in np.ndenumerate(cm):                                                #writes count into cell
        ax.text(j, i, str(v), ha="center", va="center")                                 #vertical and horizontal alignment 
    fig.colorbar(im, ax=ax)                                                             #color scale bar
    plt.savefig(filename, dpi=300, bbox_inches="tight")                                 #saves figure
    plt.show()                                                                          #displays
 
 
#-------------------------------------------------------------
#precision-recall curve plot
#-------------------------------------------------------------
def plot_pr_curve(y_true, proba, filename, thr_default=0.5, thr_tuned=None):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
 
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {ap:.3f})")
    plt.grid(True)
 
    def mark(th, label):
        if len(thresholds) == 0:
            return
        idx = np.argmin(np.abs(thresholds - th))
        plt.scatter(recall[idx + 1], precision[idx + 1], s=60)
        plt.text(recall[idx + 1], precision[idx + 1], label)
 
    mark(thr_default, f"th={thr_default}")
    if thr_tuned is not None:
        mark(thr_tuned, f"th={thr_tuned:.3f}")
 
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
 
 
#-------------------------------------------------------------
#threshold tuning  (reused from baseline script logic)
#-------------------------------------------------------------
def tune_threshold(y_true, proba, target_recall=TARGET_RECALL):                         #Finds the threshold that meets target_recall while maximising F1.
                                                                                        #Falls back to 0.5 if the target is not achievable.

    _, _, thresholds = precision_recall_curve(y_true, proba)                            #returns every possible threshold the model produces and the loop tests each one.
    best_thr, best_f1 = None, -1.0
 
    for th in thresholds:
        pred_temp = (proba >= th).astype(int)
        r  = recall_score(y_true, pred_temp, zero_division=0)
        f1 = f1_score(y_true, pred_temp, zero_division=0)
        if r >= target_recall and f1 > best_f1:                                         #recall target met AND best F1 so far
            best_f1 = f1
            best_thr = th                                                               #save this threshold as the new best
 
    if best_thr is None:
        print(f"  No threshold found with recall >= {target_recall}. Using 0.5.")       
        best_thr = 0.5
 
    return best_thr
 
 
#-------------------------------------------------------------
#STEP 1: FEATURE SELECTION
#-------------------------------------------------------------
def select_features(x_train, x_test, y_train, top_n=TOP_N_FEATS):
    
    #Two-stage feature selection:
      #Stage 1 — Variance threshold: removes near-constant columns
      #Stage 2 — XGBoost importance: keeps the top_n most predictive features
    #Returns reduced x_train, x_test, and the list of selected column names.
   
    print("\n--- Stage 1: Variance threshold ---")                                    
    var_sel    = VarianceThreshold(threshold=0.01)                                      #removes any column whose values barely change across all operating points
    x_tr_v     = var_sel.fit_transform(x_train)                                         #learns which columns to drop from training data only
    x_te_v     = var_sel.transform(x_test)                                              #applies the same filter to the test set
    kept_cols  = x_train.columns[var_sel.get_support()]                                 #True/False mask of which columns survived
    print(f"  {x_tr_v.shape[1]} / {x_train.shape[1]} features kept")
 
    print("\n--- Stage 2: XGBoost importance ranking ---")                              
    pos_weight = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))               #neg:pos ratio (~99:1) — used to up-weight risky samples
    xgb_ranker = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        verbosity=0,
        eval_metric="aucpr",
    )
    xgb_ranker.fit(x_tr_v, y_train)
 
    importances = pd.Series(
        xgb_ranker.feature_importances_, index=kept_cols
     ).sort_values(ascending=False)                                                      #rank all features by importance score, highest first
    
    top_feats = importances.head(top_n).index.tolist()                                   #keep only the top 100 column names
    print(f"  Keeping top {top_n} features")
    print(f"\n  Top 10 most important features:")
    print(importances.head(10).to_string())
 
    # --- plot top-20 importances ---
    plt.figure()
    importances.head(20)[::-1].plot(kind="barh")
    plt.xlabel("XGBoost feature importance score")
    plt.title("Top 20 selected features")
    plt.tight_layout()
    plt.savefig("xgb_fig_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()
 
    return x_train[top_feats], x_test[top_feats], top_feats
 
 
#-------------------------------------------------------------
#STEP 2: BUILD XGBOOST PIPELINE
#-------------------------------------------------------------
def build_xgb_pipeline(pos_weight):

    #XGBoost pipeline.
        #scale_pos_weight handles class imbalance (replaces class_weight="balanced")
        #SMOTE is optional
    
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),                                   #fills any missing values with the median of each column
        ("smote",   SMOTE(random_state=RANDOM_STATE)),                                   #creates synthetic risky training points so the model sees a more balanced dataset
        ("scaler",  StandardScaler()),                                                   #scales all features to similar numeric ranges
        ("clf", XGBClassifier(                                                           #the model itself. Builds 300 decision trees, each one correcting the mistakes of the previous
            
            scale_pos_weight=pos_weight,                                                 #compensates for 1% risky rate
            n_estimators=300,                                                            #number of trees more trees = more accurate
            max_depth=6,                                                                 #tree depth — controls overfitting
            learning_rate=0.05,                                                          #smaller = slower but more accurate
            subsample=0.8,                                                               #fraction of rows per tree
            colsample_bytree=0.8,                                                        #fraction of features per tree
            eval_metric="aucpr",                                                         #optimise for PR-AUC during training
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])
 
 
#-------------------------------------------------------------
#STEP 3: TRAIN, EVALUATE, PLOT
#-------------------------------------------------------------
def evaluate(model, x_tr, x_te, y_tr, y_te, label):

    #Trains the model and prints evaluation at default and tuned thresholds.
    #Saves confusion matrix and PR curve plots.
    #Returns PR-AUC and the tuned threshold value.

    model.fit(x_tr, y_tr)                                                                #trains the full pipeline 
    proba = model.predict_proba(x_te)[:, 1]                                              #gets the probability of "risky" for each test point, [:, 1] selects column 1 (risky), not column 0 (not risky).
 
    pr_auc = average_precision_score(y_te, proba)
    print(f"\n  PR-AUC: {pr_auc:.4f}")
 
    #default threshold = 0.5
    pred_05 = (proba >= 0.5).astype(int)
    print(f"\n  --- Default threshold (0.5) ---")
    print("  Confusion matrix:\n", confusion_matrix(y_te, pred_05))
    print(classification_report(y_te, pred_05, digits=4))
 
    #tuned threshold
    best_thr = tune_threshold(y_te, proba)
    pred_thr = (proba >= best_thr).astype(int)
    print(f"\n  --- Tuned threshold ({best_thr:.3f}) ---")
    print("  Confusion matrix:\n", confusion_matrix(y_te, pred_thr))
    print(classification_report(y_te, pred_thr, digits=4))
 
    #plots
    plot_confusion(
        confusion_matrix(y_te, pred_05),
        f"{label} — threshold 0.5",
        "xgb_fig_cm_default.png",
    )
    plot_confusion(
        confusion_matrix(y_te, pred_thr),
        f"{label} — threshold {best_thr:.3f}",
        "xgb_fig_cm_tuned.png",
    )
    plot_pr_curve(
        y_te, proba,
        "xgb_fig_pr_curve.png",
        thr_default=0.5,
        thr_tuned=best_thr,
    )
 
    #probability distribution
    plt.figure()
    plt.hist(proba[y_te == 0], bins=40, alpha=0.7, label="not_risky")
    plt.hist(proba[y_te == 1], bins=40, alpha=0.7, label="risky")
    plt.yscale("log")
    plt.xlabel("Predicted P(risky)")
    plt.ylabel("Count (log scale)")
    plt.title(f"Probability distribution — {label}")
    plt.legend(); plt.grid(True)
    plt.savefig("xgb_fig_proba_hist.png", dpi=300, bbox_inches="tight")
    plt.show()
 
    return pr_auc, best_thr
 
 
#-------------------------------------------------------------
#STEP 4: CROSS-VALIDATION
#-------------------------------------------------------------
def cross_validate_model(x, y, selected_features, pos_weight):
    
    #5-fold stratified cross-validation on the selected features.
    #Reports mean ± std for PR-AUC and F1
    
    print("\n--- 5-fold stratified cross-validation ---")
 
    cv_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf", XGBClassifier(
            scale_pos_weight=pos_weight,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)               #creates 5 folds, each preserving the ~1% risky rate
    scores = cross_validate(                                                                #handles the 5 train/test cycles automatically and returns an array of 5 scores.
        cv_pipeline,
        x[selected_features], y,
        cv=cv,
        scoring=["average_precision", "f1"],
        return_train_score=True,
    )
 
    print(f"  PR-AUC : {scores['test_average_precision'].mean():.4f}"
          f" ± {scores['test_average_precision'].std():.4f}")
    print(f"  F1     : {scores['test_f1'].mean():.4f}"
          f" ± {scores['test_f1'].std():.4f}")
 
 
#-------------------------------------------------------------
#MAIN
#-------------------------------------------------------------
def main():
 
    # --- load data (same file as your baseline script) ---
    df = pd.read_csv(DATA, compression="gzip")
    print("Loaded:", df.shape)
 
    y = df["risky"].astype(int)
    x = df.drop(columns=["risky"])
 
    print("Class balance:", y.value_counts().to_dict())
    print(f"Positive rate: {float(y.mean()):.4f}")
 
    # --- train/test split (80/20, stratified) ---
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
 
    # --- Calculate class ratio ---

    pos_weight = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"Class ratio (neg:pos): {pos_weight}:1")
 
    # --- feature selection ---
    #calls select_features(), runs both the variance filter and the XGBoost importance ranking, and returns the trimmed datasets.
    x_train_sel, x_test_sel, selected_features = select_features(
        x_train, x_test, y_train
    )
 
    # sampling_strategy = ratio of minority:majority after SMOTE
    # None = no SMOTE (rely on scale_pos_weight only)
    # 0.1  = 1:10 ratio,  0.2 = 1:5 ratio,  1.0 = fully balanced
    SMOTE_RATIOS = {
        "Natural (no SMOTE)" : None,
        "1:10  (0.1)"        : 0.1,
        "1:5   (0.2)"        : 0.2,
        "1:1   (1.0)"        : 1.0,
    }

    ratio_results = {}

    for label, ratio in SMOTE_RATIOS.items():

        # build pipeline with this ratio
        steps = [("imputer", SimpleImputer(strategy="median"))]

        if ratio is not None:
            steps.append(("smote", SMOTE(
                sampling_strategy=ratio,
                random_state=RANDOM_STATE
            )))

        steps += [
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                scale_pos_weight=pos_weight,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
            ))
        ]

        m = Pipeline(steps)
        m.fit(x_train_sel, y_train)
        proba = m.predict_proba(x_test_sel)[:, 1]

        pr_auc  = average_precision_score(y_test, proba)
        thr     = tune_threshold(y_test, proba)
        pred    = (proba >= thr).astype(int)
        cm      = confusion_matrix(y_test, pred)

        tn = cm[0,0]; fp = cm[0,1]
        fn = cm[1,0]; tp = cm[1,1]
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        ratio_results[label] = {
            "PR-AUC"    : pr_auc,
            "Threshold" : thr,
            "Recall"    : recall,
            "Precision" : precision,
            "FP"        : fp,
            "FN"        : fn,
        }

        print(f"\n{label}: PR-AUC={pr_auc:.4f}  Recall={recall:.3f}  FP={fp}  FN={fn}")

        # confusion matrix printout for this ratio
        print(f"\n  Confusion matrix — {label}")
        print(confusion_matrix(y_test, pred))

    # --- summary comparison table ---
    print("\n" + "="*70)
    print(f"  {'Ratio':<20} {'PR-AUC':>8} {'Thr':>7} {'Recall':>8} {'Prec':>8} {'FP':>5} {'FN':>5}")
    print("-"*70)
    for name, r in ratio_results.items():
        print(f"  {name:<20} {r['PR-AUC']:>8.4f} {r['Threshold']:>7.3f}"
              f" {r['Recall']:>8.3f} {r['Precision']:>8.3f} {r['FP']:>5} {r['FN']:>5}")
    print("="*70)

    # --- build and evaluate XGBoost ---
    print("\n" + "="*55)                            #repeat the (=) character 55 times
    print("  XGBoost — top 100 selected features")
    print("="*55)
    model = build_xgb_pipeline(pos_weight)          #Builds model
    pr_auc, best_thr = evaluate(
        model,                                      #Training begins
        x_train_sel, x_test_sel,                    #trimmed datasets
        y_train, y_test,
        label="XGBoost (selected features)",
    )
 
    # --- cross-validation ---
    cross_validate_model(x, y, selected_features, pos_weight)
 
    # --- final summary ---
    print("\n" + "="*55)            
    print("  SUMMARY")
    print("="*55)
    print(f"  Features used  : {len(selected_features)} / {x.shape[1]}")        #lens(): counts how many features were selected / {x.shape[1]}: total number of columns
    print(f"  PR-AUC         : {pr_auc:.4f}")
    print(f"  Tuned threshold: {best_thr:.3f}")
    print(f"  Target recall  : {TARGET_RECALL}")
    print("="*55)
 
 
if __name__ == "__main__":
    main()