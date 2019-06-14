from predict import predict
from utils.load import load_features_group, load_label_df
from utils.plotting import print_confusion_matrix

import sys
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt



if __name__ == "__main__":
    feature_files = sys.argv[1:-1]
    label = sys.argv[-1]
    result = predict(feature_files)
    label_df = load_label_df(label)

    pred_and_true = result.merge(label_df, on="bookingID")
    pred = pred_and_true['prediction']
    true = pred_and_true['label']
    pred_proba = pred_and_true['positive_prediction_probability']

    print("Accuracy Score:", accuracy_score(true, pred))
    print("Recall Score:", recall_score(true, pred))
    print("Precision Score:", precision_score(true, pred))
    print("F1 Score:", f1_score(true, pred))
    print("AUC:", roc_auc_score(true, pred_proba))

    cm = confusion_matrix(true, pred)
    print_confusion_matrix(cm, [0,1])
    plt.show()