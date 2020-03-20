import argparse
import os

from src.infer import infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the folder containing the test images')
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--output_dir', default='output',
                        help='Path to the output folder where the predictions will be saved')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    os.makedirs(args.output_dir, exist_ok=True)

    test_files = []
    for root, dirs, files in os.walk(args.data_path):
        for file_name in files:
            if file_name[-4:] == ".png":
                test_files.append(os.path.join(root, file_name))

    infer(test_files, args.checkpoint_path, args.output_dir, 16)


if __name__ == '__main__':
    main()
