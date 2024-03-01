import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


def get_dataloader(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (args.image_size*2, args.image_size*2), antialias=True
            ),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomCrop((args.image_size, args.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
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
