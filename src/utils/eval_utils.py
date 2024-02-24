import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dvclive import Live
from src.pylogger import logger







def get_metrics(train_test_dir_path,
                model_file_path,
                metrics_file_plot_path,
                target_column):

    test_data = pd.read_csv(train_test_dir_path/"test.csv", encoding = "ISO-8859-1")
    model = joblib.load(model_file_path)
    logger.info(f"load model and test data {model_file_path}")

    test_x = np.array(test_data.drop([target_column], axis=1))
    test_y = np.array(test_data[[target_column]])

    predicted_booking = model.predict(test_x)
    predicted_booking = predicted_booking.reshape(predicted_booking.shape[0], 1)
    logger.info(f'Predicted bookings:{target_column}')
    print(test_y.shape, predicted_booking.shape)
    accuracy = accuracy_score(test_y, predicted_booking)
    precision = precision_score(test_y, predicted_booking)
    recall = recall_score(test_y, predicted_booking)
    f1 = f1_score(test_y, predicted_booking)
    scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    logger.info(f"Calculating Model Scores")
    # Plot confusion matrix
    cm = confusion_matrix(test_y, predicted_booking)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save confusion matrix
    plt.savefig(metrics_file_plot_path / "confusion_matrix.png", dpi=150, bbox_inches='tight', pad_inches=0)
    logger.info("Save confusion matrix")

    with Live(dir="dvcliveevaluation",report="md") as live:
        #live.log_params(params=params)
        
        live.log_metric("accuracy", scores["accuracy"])
        live.log_metric("precision", scores["precision"])
        live.log_metric("recall", scores["recall"])
        live.log_metric( "f1", scores["f1"])
        live.log_sklearn_plot("confusion_matrix", test_y, predicted_booking)
    return scores
