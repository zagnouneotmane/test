import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.data_utils import create_test_dataset
from src.utils.load_params import load_params


def data_split(params):
    random_state = params.base.random_state
    data_dir = Path(params.data_load.data_dir)
    file_name = params.data_load.file_name
    file_path = data_dir/file_name


    train_test_dir_path = Path(params.data_split.train_test_dir_path)
    
    train_test_dir_path.mkdir(exist_ok=True)
    test_pct = params.data_split.test_pct

    create_test_dataset(file_path,
                        train_test_dir_path,
                        test_pct,
                        random_state)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    data_split(params)
