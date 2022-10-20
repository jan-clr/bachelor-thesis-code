import sys
import os
import cv2
import numpy as np


def normalize_data(path) -> str:
	"""
	Normalizes all image files in a provided directory and saves the normalized copies to path + '_normalized'

	:param path: The path of the directory with the files to be normalized.
	:return: The path to the normalized images
	"""
	if not os.path.isdir(path):
		print('Path is not a directory')
		quit()

	images = []

	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		parts = file.split('.')
		name, ext = parts[0], parts[1]
		if os.path.isfile(file_path):
			images.append({'path': file_path, 'ext': ext, 'name': name})

	print(f"Found {len(images)} images.")

	norm_dir = f"{path}_normalized"
	if not os.path.isdir(norm_dir):
		os.mkdir(norm_dir)

	nr_norm = 0
	print("")
	for img in images:
		cv_img = cv2.imread(img['path'])
		if cv_img is not None:
			img_norm = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX)
			# cv2.imshow('Test', img_norm)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			cv2.imwrite(f"{norm_dir}/{img['name']}_normalized.{img['ext']}", img_norm)
			nr_norm += 1
			print(f"\rNormalized {nr_norm}/{len(images)}", end='')

	print(f"\nNormalized images saved to '{norm_dir}'")

	return norm_dir


def load_imgs(path):
	if not os.path.isdir(path):
		print('Path is not a directory')
		quit()

	files = []

	for file in os.listdir(path):
		file_path = os.path.join(path, file)
		parts = file.split('.')
		name, ext = parts[0], parts[1]
		if os.path.isfile(file_path):
			files.append({'path': file_path, 'ext': ext, 'name': name})

	images = []

	for i, img in enumerate(files):
		print(i, img['name'])
		cv_img = cv2.imread(img['path'])
		if cv_img is not None:
			images.append(cv_img)

	images = np.array(images)
	return images


def main():
	if len(sys.argv) != 2:
		print('Please provide a path to the folder with the images you wish to be normalized.')
		quit()

	path = sys.argv[1]

	#normalize_data(path)
	images = load_imgs(path).astype('int32')

	mean_img = np.mean(images, axis=0)
	mean_img = mean_img.astype('uint8')
	mean_val = np.mean(images)

	images = (images - mean_img + mean_val)
	min_val = np.min(images)
	max_val = np.max(images)
	images = ((images - min_val) / (max_val - min_val) * 255.0).astype('uint8')

	#print(images[0])
	#for i, img in enumerate(images):
	#	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#	cv2.imwrite(os.path.join(path, f"{i}.tif"), img)


if __name__ == '__main__':
	main()
