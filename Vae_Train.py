import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Variational_Auto_Encoding_Bayes import VAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)


SAVE_DIR = "/content/drive/MyDrive/Vae_ckpt"

os.makedirs(SAVE_DIR, exist_ok=True)

print("save dir:", SAVE_DIR)


BATCH_SIZE = 2048

LR = 8.48528137423857e-3

EPOCHS = 300


transform = transforms.ToTensor()

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def train_epoch(model, loader, optimizer, beta):
    model.train()

    total = 0.0

    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)

        y, mu, logvar = model(x)

        loss = model.loss(x, y, mu, logvar, beta=beta)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()

    total = 0.0

    for x, _ in loader:
        x = x.to(device).view(x.size(0), -1)

        y, mu, logvar = model(x)

        total += model.loss(x, y, mu, logvar).item() * x.size(0)

    return total / len(loader.dataset)


model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

best_test = float("inf")


for ep in range(1, EPOCHS + 1):
    beta = min(1.0, ep / 10.0)

    tr = train_epoch(model, train_loader, optimizer, beta)

    te = eval_epoch(model, test_loader)

    print(f"[{ep:02d}/{EPOCHS}] train={tr:.2f} test={te:.2f} beta={beta:.2f}")

    torch.save({
        "epoch": ep,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": tr,
        "test_loss": te,
        "beta": beta,
    }, f"{SAVE_DIR}/vae_last.pt")

    if te < best_test:
        best_test = te

        torch.save(model.state_dict(), f"{SAVE_DIR}/vae_best.pt")

        print(f"best saved test={te:.2f}")


print(f"done best_test={best_test:.2f}")

print("files:", os.listdir(SAVE_DIR))