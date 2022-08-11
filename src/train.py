import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from transforms import transform
from cityscapes_dataset import CustomCityscapesDataset
from model import CS_UNET
from utils import save_checkpoint, load_checkpoint


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
CS_DATA_DIR = '../data/Cityscapes'


def show_img_and_pred(img, target=None, prediction=None):
    fig = plt.figure()

    total_plots = sum(x is not None for x in [img, target, prediction])
    i = 1

    # show image
    ax = plt.subplot(1, total_plots, i)
    plt.title('Image')
    plt.imshow(img[0].permute(1, 2, 0))
    i += 1
    # show mask
    if target is not None:
        ax = plt.subplot(1, total_plots, i)
        plt.title('Target')
        plt.imshow(target[0])
        i += 1
    # show prediction
    if prediction is not None:
        ax = plt.subplot(1, total_plots, i)
        plt.title('Prediction')
        plt.imshow(prediction[0])
        i += 1
    plt.show()


def train_loop_CS(loader, model, optimizer, loss_fn):
    size = len(loader.dataset)
    for batch, (X, y) in enumerate(loader):
        X = X.float().to(DEVICE)
        # y is still long for some reason
        y = y.to(DEVICE)

        # Compute prediction and loss
        #pred = torch.argmax(torch.softmax(model(X), dim=1), dim=1)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop_CS(loader, model, loss_fn):
    pass


def main():

    train_data = CustomCityscapesDataset(CS_DATA_DIR, transforms=transform)
    test_data = CustomCityscapesDataset(CS_DATA_DIR, mode='val', transforms=transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    model = CS_UNET(in_ch=3, out_ch=len(train_data.classes)).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print("\nBeginning Training\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop_CS(loader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss_fn)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, 'first_try.pth.tar')

    print("\nTraining Complete.")


if __name__ == '__main__':
    main()
