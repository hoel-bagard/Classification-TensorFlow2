import argparse
import os
import shutil
import glob

import tensorflow as tf

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.dataset.dataset_creator import DatasetCreator
from src.networks.cnn import CNN
from src.networks.small_mobile_net import SmallMobileNet
from src.networks.mobile_net import MobileNetV2
from src.train import train


def main():
    parser = argparse.ArgumentParser("MNIST - TensorFlow 2")
    parser.add_argument("--pickle", action="store_true", help="Loads / stores data to pickle for faster loading")
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if not DataConfig.KEEP_TB:
        shutil.rmtree(DataConfig.TB_DIR, ignore_errors=True)
        while os.path.exists(DataConfig.TB_DIR):
            pass
    os.makedirs(DataConfig.TB_DIR, exist_ok=True)

    if DataConfig.USE_CHECKPOINT:
        if not DataConfig.KEEP_CHECKPOINTS:
            shutil.rmtree(DataConfig.CHECKPOINT_DIR, ignore_errors=True)
            while os.path.exists(DataConfig.CHECKPOINT_DIR):
                pass
        os.makedirs(DataConfig.CHECKPOINT_DIR, exist_ok=True)

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        output_folder = os.path.join(DataConfig.CHECKPOINT_DIR, "MNIST-TensorFlow")
        for filepath in glob.glob(os.path.join("**", "*.py"), recursive=True):
            destination_path = os.path.join(output_folder, filepath)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(filepath, destination_path)
        misc_files = ["README.md", "requirements.txt", "setup.cfg"]
        for misc_file in misc_files:
            shutil.copy(misc_file, os.path.join(output_folder, misc_file))
        print("Finished copying files")

    dataset = DatasetCreator(DataConfig.DATA_PATH, DataConfig.DATASET, batch_size=ModelConfig.BATCH_SIZE,
                             cache=False, pickle=args.pickle)

    if ModelConfig.NETWORK_NAME == "CNN":
        model = CNN(dataset.input_shape, dataset.classes_nb)
    elif ModelConfig.NETWORK_NAME == "SmallMobileNet":
        model = SmallMobileNet(dataset.input_shape, dataset.classes_nb)
    elif ModelConfig.NETWORK_NAME == "MobileNetV2":
        model = MobileNetV2(dataset.input_shape, dataset.classes_nb)
    model.build((None, *dataset.input_shape))

    print(model.summary())

    train(model, dataset)


if __name__ == "__main__":
    main()
