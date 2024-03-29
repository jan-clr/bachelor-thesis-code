import cv2

from src.detection import load_image
from glob import glob


def main():
    label_map = {2: 1, 3: 0, 4: 0}
    placeholder = 200
    root = 'data/vapourbase_binary/gtFine'
    files = glob(f"{root}/**/*labelIds*", recursive=True)
    for file in files:
        img = load_image(file)
        for old_label, new_label in label_map.items():
            img[img == old_label] = new_label + placeholder
        for old_label, new_label in label_map.items():
            img[img == new_label + placeholder] = new_label
        cv2.imwrite(filename=file, img=img)


if __name__ == '__main__':
    main()
