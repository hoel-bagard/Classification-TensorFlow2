import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser("CatVsDog dataset splitting")
    parser.add_argument("data_path", type=str, help="Path to the root folder of the dataset")
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "Train")
    val_path = os.path.join(args.data_path, "Validation")

    os.makedirs(val_path, exist_ok=True)

    # Split 80% in train and 20% in validation
    for i in range(8000, 10000, 1):
        print(f"Moving image {i}     ", end="\r")
        # Move cat image
        src = os.path.join(train_path, f"cat.{i}.jpg")
        dst = os.path.join(val_path, f"cat.{i}.jpg")
        shutil.move(src, dst)

        # Move dog image
        src = os.path.join(train_path, f"dog.{i}.jpg")
        dst = os.path.join(val_path, f"dog.{i}.jpg")
        shutil.move(src, dst)

    print("\nFlinished splitting dataset")


if __name__ == "__main__":
    main()
