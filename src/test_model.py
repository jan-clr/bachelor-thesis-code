import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from transforms import transforms_train
from datasets import CustomCityscapesDataset
from model import UnetResEncoder
from utils import IoU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = True
ROOT_DATA_DIR = '../data'
DATASET_NAME = 'Cityscapes'


def single_batch_overfit(X, y, model, optimizer, loss_fn):
    X = X.float().to(DEVICE)
    # y is still long for some reason
    y = y.to(DEVICE)
    # print(y.min(), y.max(), torch.unique(y))
    for epoch in range(NUM_EPOCHS):

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        jaccard_idx, scores = IoU(pred=torch.argmax(nn.functional.softmax(pred, 1), 1), ground_truth=y, n_classes=19)
        print(f"Epoch {epoch + 1} - Jaccard: {jaccard_idx} loss: {loss.item()}")


def main():
    run_name = f"no_aug_SGD"  # _{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}
    run_dir = f"../runs/{DATASET_NAME}/{run_name}"
    run_file = f"{run_dir}/model.pth.tar"

    current_dataset = DATASET_NAME
    data_dir = f"{ROOT_DATA_DIR}/{current_dataset}"

    test_data = CustomCityscapesDataset(data_dir, mode='train', transforms=transforms_train)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    model = UnetResEncoder(in_ch=3, out_ch=len(test_data.classes)).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X, y = iter(test_dataloader).next()

    single_batch_overfit(X,y, model, optimizer, loss_fn)

    """
    X = X[0].float().to(DEVICE).unsqueeze(0)
    y = y[0].to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        pred = model(X)
        loss = loss_fn(pred, y)
        pred = torch.argmax(nn.functional.softmax(pred.float(), 1), 1)
        jaccard_idx, scores = IoU(pred=pred, ground_truth=y,
                                  n_classes=len(test_dataloader.dataset.classes))
        print(y, pred)
        print(loss, jaccard_idx)
        # show_img_and_pred(X.cpu(), y.cpu(), pred.cpu())
    """
    # losses, ious = val_fn(loader=test_dataloader, model=model, loss_fn=loss_fn)
    #
    # print(losses)
    # print(ious)
    #
    # print(f"Loss Avg {np.array(losses).sum() / len(losses)}")
    # print(f"Jaccard Avg {np.array(ious).sum() / len(ious)}")


if __name__ == '__main__':
    main()