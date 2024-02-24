from src.pylogger import logger
import pandas as pd
from dvclive import Live
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import learning_curve
import joblib
import os
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif



def train_model(train_test_dir_path,
                selectkbest__k,
                n_estimators,
                class_weight,
                max_depth,
                criterion,
                min_samples_split,
                max_features,
                min_samples_leaf,
                target_column,
                model_name,
                model_path,
                seed):
    

    train_data = pd.read_csv(train_test_dir_path/ 'train.csv', encoding = "ISO-8859-1")


    train_x = np.array(train_data.drop([target_column], axis=1))
    train_y = np.array(train_data[[target_column]])
    train_y = np.ravel(train_y)
    np.seterr(divide='ignore', invalid='ignore')

    rf_classifier  = RandomForestClassifier(n_estimators=n_estimators,
                                                class_weight=class_weight,
                                                max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                max_features=max_features,
                                                criterion=criterion,
                                                min_samples_leaf=min_samples_leaf,
                                                random_state=seed)
    model = make_pipeline(SelectKBest(f_classif, k=selectkbest__k), rf_classifier )
    model = rf_classifier
    model.fit(train_x, train_y)
    logger.info(f"training model {model_name}")
    live = Live(dir="dvclivetraining")
    train_sizes=np.linspace(0.05, 1, 20)
    N, train_score, val_score = learning_curve(model, train_x, train_y,
                                              cv=5, scoring='f1',
                                               train_sizes=train_sizes)
    print(N)
    metrics={"train_score":{},"val_score":{}, "step":{}}
    #for i, train_sizes in enumerate(N):
    for i, train_sizes in enumerate(train_sizes):
                
                metrics["train_score"][i] = train_score.mean(axis=1)[i]
                metrics["val_score"][i] = val_score.mean(axis=1)[i]
                metrics["step"][i] = i
                live.log_metric(f"train_score", train_score.mean(axis=1)[i])
                live.log_metric(f"val_score", val_score.mean(axis=1)[i])
                live.next_step()

    
    logger.info(f"DVCLive {learning_curve}")    

    joblib.dump(model, os.path.join(model_path, model_name))
    logger.info(f"Downloading model {model_name} into file {model_path/model_name}")
    return metrics