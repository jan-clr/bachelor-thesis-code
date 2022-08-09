import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transforms import transform

from cityscapes_dataset import CustomCityscapesDataset


def main():
    root_dir = '../data/Cityscapes'

    train_data = CustomCityscapesDataset(root_dir, transforms=transform)
    test_data = CustomCityscapesDataset(root_dir, mode='val', transforms=transform)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    img, target = next(iter(train_dataloader))

    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0))
    ax = plt.subplot(1, 2, 2)
    plt.imshow(target[0])
    plt.show()


if __name__ == '__main__':
    main()