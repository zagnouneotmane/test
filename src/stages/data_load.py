import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.data_utils import dataset_prep
from src.utils.load_params import load_params
from dotenv import load_dotenv


    


def data_load(params):

    data_dir = Path(params.data_load.data_dir)
    data_dir.mkdir(exist_ok=True)
    #orig_dirname = params.data_load.orig_dirname
    file_name = params.data_load.file_name
    dataset_prep(data_dir=data_dir,
                 file_name=file_name)
    


if __name__ == '__main__':
    load_dotenv()
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    data_load(params)
