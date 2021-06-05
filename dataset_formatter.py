import pandas as pd
import numpy as np
import cv2
from PIL import Image

def get_animal_array(animal_name):
	animal_list = []
	for i in range(0, 12499):
		img = cv2.imread(f"images/train/{animal_name}.{i}.jpg")

		# Change to RBG as cv2 assumes BGR
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# Skip image if they are not at least 128x128
		if(img.shape[0] < 128 or img.shape[1] < 128):
			continue

		# Resize images to 128x128
		img_resized = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
		im = Image.fromarray(img_resized, mode = 'RGB')
		animal_list.append(img_resized)

	return np.stack(animal_list)

def main():
	dog_array = get_animal_array("dog")
	dog_labels = np.ones(dog_array.shape[0])

	print("Dog array shape: " + str(dog_array.shape))
	print("Dog labels shape: " + str(dog_labels.shape))

	print("-------------------------------------------")

	cat_array = get_animal_array("cat")
	cat_labels = np.zeros(cat_array.shape[0])

	print("Cat array shape: " + str(cat_array.shape))
	print("Cat labels shape: " + str(cat_labels.shape))

	print("-------------------------------------------")

	animal_array = np.concatenate((dog_array, cat_array), axis = 0)
	animal_labels = np.concatenate((dog_labels, cat_labels), axis = 0)

	print("Animal array shape: " + str(animal_array.shape))
	print("Animal labels shape: " + str(animal_labels.shape))

	# Shuffling both arrays and label
	random_order = np.arange(len(animal_array))
	np.random.shuffle(random_order)

	animal_array = animal_array[random_order]
	animal_labels = animal_labels[random_order]

	np.save("x_train.npy", animal_array)
	np.save("y_train.npy", animal_labels)


if __name__ == "__main__":
	main()