import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))
import os
import argparse
import json

from src.utils.eval_utils import get_metrics
from src.utils.load_params import load_params


def evaluate(params):
    train_test_dir_path = Path(params.data_split.train_test_dir_path)
    target_column = params.train.target_column
    model_path = params.train.model_path
    model_name = params.train.model_name
    model_file_path = Path(os.path.join(model_path, model_name)).absolute()
    metrics_file_path = params.evaluate.metrics_file
    metrics_file_plot_path = Path(params.evaluate.metrics_file).parent
    Path(params.evaluate.metrics_file).parent.mkdir(parents=True, exist_ok=True)
    metrics = get_metrics(train_test_dir_path=train_test_dir_path,
                          model_file_path=model_file_path,
                          metrics_file_plot_path=metrics_file_plot_path,
                          target_column=target_column)
    Path(params.evaluate.metrics_file).parent.mkdir(parents=True, exist_ok=True)

    json.dump(
        obj=metrics,
        fp=open(metrics_file_path, 'w'),
        indent=4
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    evaluate(params)
