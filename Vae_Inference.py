import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets

from Variational_Auto_Encoding_Bayes import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = os.path.join(os.path.expanduser("~"), "ai_project/my_library/Gen/Vae_Vanilla")

CKPT_PATH = "vae_last.pt"

LATENT_DIM = 20

HIDDEN_DIM = 400


model = VAE(input_dim=784, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)

state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

model.load_state_dict(state_dict)

model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


@torch.no_grad()

def reconstruct(img_tensor):

    x = img_tensor.view(1, -1).to(device)

    y, mu, logvar = model(x)

    return y.view(28, 28).cpu().numpy()


def preprocess_path(path):

    img = Image.open(path).convert("L").resize((28, 28))

    arr = np.array(img, dtype=np.float32) / 255.0

    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)

    return torch.from_numpy(arr).unsqueeze(0)


def reconstruct_from_path(path, save_path=None):

    img_t = preprocess_path(path)

    recon = reconstruct(img_t)

    fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))

    ax[0].imshow(img_t.squeeze(), cmap="gray")

    ax[0].set_title("input")

    ax[0].axis("off")

    ax[1].imshow(recon, cmap="gray")

    ax[1].set_title("recon")

    ax[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)

        print("saved:", save_path)

    plt.show()

    return recon


@torch.no_grad()

def reconstruct_from_testset(n=8, save_path=None):

    test_ds = datasets.MNIST(root="./data", train=False, download=False, transform=transforms.ToTensor())

    fig, ax = plt.subplots(2, n, figsize=(n * 1.2, 3))

    for i in range(n):
        img_t, _ = test_ds[i]

        recon = reconstruct(img_t)

        ax[0, i].imshow(img_t.squeeze(), cmap="gray")

        ax[0, i].axis("off")

        ax[1, i].imshow(recon, cmap="gray")

        ax[1, i].axis("off")

    ax[0, 0].set_ylabel("input")

    ax[1, 0].set_ylabel("recon")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)

        print("saved:", save_path)

    plt.show()


@torch.no_grad()
def sample_from_prior(n=8, save_path=None):
    model.eval()

    z = torch.randn(n, LATENT_DIM, device=device)

    y = model.decoder(z).view(-1, 28, 28).cpu().numpy()

    fig, ax = plt.subplots(1, n, figsize=(n * 1.2, 1.5))

    for i in range(n):
        ax[i].imshow(y[i], cmap="gray")

        ax[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)

        print("saved:", save_path)

    plt.show()


if __name__ == "__main__":
    reconstruct_from_testset(n=8, save_path=f"{SAVE_DIR}/recon_grid.png")

    sample_from_prior(n=8, save_path=f"{SAVE_DIR}/samples.png")

    #reconstruct_from_path(".png", save_path=f"{SAVE_DIR}/recon_single.png")