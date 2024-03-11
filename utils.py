import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from glob import glob
import os

def check_Image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


class CelebDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.imgs_paths = glob(os.path.join(path, "*.jpg"))
        self.len = len(self.imgs_paths)
        self.transforms = transforms

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = Image.open(self.imgs_paths[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, None  # None is the dummy class not using


def get_dataloader(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (args.image_size, args.image_size), antialias=True
            ),  # args.image_size + 1/4 *args.image_size
            # torchvision.transforms.RandomCrop((args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataset = CelebDataset(args.dataset_path, transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def save_checkpoint(
    model_state,
    optim_state=None,
    epoch=None,
    losses=None,
    filename="checkpoint.pt",
):
    state = {
        "model_state": model_state,
    }
    if optim_state is not None:
        state["optim_state"] = optim_state
    if epoch is not None:
        state["epoch"] = epoch
    if losses is not None:
        state["losses"] = losses

    torch.save(state, filename)
