from src.utils import normalize_data
import sys


def main():
	if len(sys.argv) != 2:
		print('Please provide a path to the folder with the images you wish to be normalized.')
		quit()

	path = sys.argv[1]

	normalize_data(path)


if __name__ == '__main__':
	main()
