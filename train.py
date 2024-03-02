import torch
from torchvision.utils import save_image
from accelerate import Accelerator
import argparse
import os

from ddpm import Diffusion
from unet import UNet
from utils import get_dataloader, save_checkpoint

accelerator = Accelerator()
device = torch.device("cuda")

SAVE_DIR = "./progress"


def main(args):
    print("Loading model...")
    model = UNet().to(device)
    print("Model loaded!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("Building diffusion class...")
    diffusion = Diffusion(img_size=args.image_size, device=device)
    print("Done!")
    crit = torch.nn.MSELoss().to(device)
    print("Fetching dataloader...")
    dataloader = get_dataloader(args)
    print("Done!")
    
    n_updates = 0
    losses = []
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch}/{args.max_epoch} : ", end="")
        running_loss = 0
        for images, labels in dataloader:
            model.train()
            images = images.to(device)
            t = diffusion.sample_timesteps(args.batch_size).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            predicted_noise = model(x_t, t)
            loss = crit(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        lossess.append(running_loss / len(dataloader))
        print(f"Loss = {losses[-1]}")
        if epoch % 50 == 0:
            sample_images = diffusion.sample(model, n=args.batch_size)
            os.makedirs(SAVE_DIR, exist_ok=True)
            save_image(sample_images, os.path.join(SAVE_DIR, f"epoch_{epoch}.jpg"))
            save_checkpoint(
                model.state_dict(), optimizer.state_dict(), epoch + 1, losses
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Path to the dataset to be trained on"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=100,
        help="Max no. of epochs to train the model on",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
