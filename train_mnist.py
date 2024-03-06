import torch
from torchvision.utils import save_image
import argparse
import os
from tqdm import tqdm

from ddpm import Diffusion
from nn import UNetModel
from utils import save_checkpoint

device = torch.device("cuda")

SAVE_DIR = "./progress_mnist"


def main(args):
    print("Loading model...")
    model = UNetModel(
        args.in_channels,
        args.model_channel,
        args.out_channels,
        args.n_resblocks,
        args.n_heads,
        args.groups,
        args.dropout,
        n_classes=None,
    ).to(device)
    print("Model loaded!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("Building diffusion class...")
    diffusion = Diffusion(img_size=args.image_size, device=device)
    print("Done!")
    crit = torch.nn.MSELoss().to(device)
    print("Fetching dataloader...")
    # dataloader = get_dataloader(args)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_ds = MNIST("./MNIST", download=True, train=True, transform=transform)
    test_ds = MNIST("./MNIST", download=False, train=False, transform=transform)

    mnist_ds = ConcatDataset([train_ds, test_ds])
    dataloader = DataLoader(mnist_ds, batch_size=args.batch_size, shuffle=True)
    print("Done!")

    losses = []
    for epoch in range(args.max_epoch):
        # print(f"Epoch {epoch}/{args.max_epoch} : ", end="")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.max_epoch} : ")
        for images, labels in pbar:
            model.train()
            images = images.to(device)
            t = diffusion.sample_timesteps(images.size(0)).to(
                device
            )  # last batch can be uneven
            x_t, noise = diffusion.noise_image(images, t)
            predicted_noise = model(x_t, t)
            loss = crit(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1

            # running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

            # lossess.append(running_loss / len(dataloader))
            # print(f"Loss = {losses[-1]}")
            if epoch % 50 == 0:
                sample_images = diffusion.sample(model, n=64)
                os.makedirs(SAVE_DIR, exist_ok=True)
                save_image(
                    sample_images, os.path.join(SAVE_DIR, f"updates_{updates}.jpg")
                )

                save_checkpoint(
                    model.state_dict(),
                    optimizer.state_dict(),
                    # epoch + 1,
                    filename="checkpoint_mnist.pt"
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
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--max_epoch",
        type=None,
        default=None,
        help="Max no. of epochs to train the model on",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model_channel",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_resblocks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=32,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
