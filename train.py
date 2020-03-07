import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

import src.network as network

def load_data(data_path):
	imgs = []
	labels = []
	for i in range(10):
		for entry in glob.glob(os.path.join(data_path, "training", str(i), "*.png"), recursive=True):
			print("Loading data {}".format(entry), end="\r")
			image = Image.open(entry)
			imgs.append(np.asarray(image))
			labels.append(i)
	try:
	    print('Data loaded' + str(' ' * (os.get_terminal_size()[0] - 11)))
	except:
		print("Data loaded" + ' '*40)
	return np.asarray(imgs), np.asarray(labels)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", help="Path to the data folder")
	parser.add_argument("--pickle", help="Path to the pickle file")
	parser.add_argument("--output_dir", default="output", help="Output directory")
	arguments = parser.parse_args()

	if arguments.data_path:
		imgs, labels = load_data(arguments.data_path)
		if arguments.pickle:
			os.makedirs(os.path.dirname(arguments.pickle), exist_ok=True)
			np.save(arguments.pickle + "imgs.npy", imgs)
			np.save(arguments.pickle + "labels.npy", labels)
	elif arguments.pickle:
		imgs = np.load(arguments.pickle + "imgs.npy")
		labels = np.load(arguments.pickle + "labels.npy")

	print("Loaded {} data of shape {}".format(len(imgs), imgs.shape[1:]))

	network.train(imgs, labels, arguments.output_dir, 50, 128)


if __name__ == "__main__":
	main()