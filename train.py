import torch
from torchvision.utils import save_image
from accelerate import Accelerator
import argparse
import os
from tqdm import tqdm

from ddpm import Diffusion

# from unet import UNet
from nn import UNetModel
from utils import get_dataloader, save_checkpoint

accelerator = Accelerator()
# device = torch.device("cuda")

SAVE_DIR = "./progress_celeb"


def main(args):
    accelerator.print("Loading model...")
    # model = UNet()
    model = UNetModel(
        in_channels=3,
        model_channel=32,
        out_channels=3,
        n_resblocks=2,
        n_heads=8,
        groups=32,
        dropout=0.5,
        channel_mult=(1, 8, 16),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.99),
    )
    # model.half()
    # model.to(device)
    accelerator.print("Model loaded!")
    accelerator.print("Building diffusion class...")
    diffusion = Diffusion(img_size=args.image_size)
    accelerator.print("Done!")
    crit = torch.nn.MSELoss()
    accelerator.print("Fetching dataloader...")
    dataloader = get_dataloader(args)
    accelerator.print("Done!")
    start = 0
    n_updates = 0
    losses = []
    updates = 0
    # scaler = torch.cuda.amp.GradScaler()
    model, diffusion, optimizer, dataloader = accelerator.prepare(
        model, diffusion, optimizer, dataloader
    )
    for epoch in range(start + 1, args.max_epoch):
        # accelerator.print(f"Epoch {epoch}/{args.max_epoch} : ", end="")
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{args.max_epoch} : ",
            disable=not accelerator.is_local_main_process,
        )
        for images, labels in pbar:
            model.train()
            # images = images.to(torch.half)
            # images = images.to(device)
            t = diffusion.sample_timesteps(images.size(0))  # last batch can be uneven
            x_t, noise = diffusion.noise_image(images, t)
            # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            predicted_noise = model(x_t, t)
            loss = crit(predicted_noise, noise)

            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            updates += 1

            # running_loss += loss.item()
            # pbar.set_postfix(loss=loss.item())

            # lossess.append(running_loss / len(dataloader))
            # accelerator.print(f"Loss = {losses[-1]}")
        accelerator.wait_for_everyone()
        if epoch % 10 == 0 and accelerator.is_local_main_process:
            accelerator.print("Checkpoint reached")
            sample_images = diffusion.sample(
                model, n=64, img_size=args.image_size, device=accelerator.device
            )
            os.makedirs(SAVE_DIR, exist_ok=True)
            save_image(sample_images, os.path.join(SAVE_DIR, f"updates_{updates}.jpg"))
            # accelerator.save_state("checkpoint_celeb.pt")
            save_checkpoint(
                model.state_dict(),
                optimizer.state_dict(),
                epoch + 1,
                filename="checkpoint_celeb.pt",
                # accelerator=accelerator,
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
        default=None,
        help="Max no. of epochs to train the model on",
    )
    # parser.add_argument(
    #     "--max_updates",
    #     type=int,
    #     default=100,
    #     help="Max amount of updates to apply to the model",
    # )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        help="load checkpoint",
        default="",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.cuda.manual_seed_all(0)
    args = get_args()
    main(args)
