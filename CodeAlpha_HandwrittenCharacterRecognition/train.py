import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from model import CharacterCNN


def fix_emnist_orientation(img):
    """Rotate/flip EMNIST samples into normal upright writing orientation."""
    return TF.hflip(TF.rotate(img, -90))


class RelabelDataset(Dataset):
    """Wrap a dataset and remap its integer class labels."""

    def __init__(self, base, subtract=0, offset=0):
        self.base = base
        self.subtract = subtract
        self.offset = offset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y) - self.subtract + self.offset


def build_datasets(kind):
    # MNIST is already upright. EMNIST is stored in a rotated/flipped orientation,
    # so correct it here to match normal handwritten uploads in the Streamlit app.
    mnist_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    emnist_tfm = transforms.Compose([
        transforms.Lambda(fix_emnist_orientation),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if kind == "mnist":
        full = datasets.MNIST("data", train=True, download=True, transform=mnist_tfm)
        test = datasets.MNIST("data", train=False, download=True, transform=mnist_tfm)
        labels = [str(i) for i in range(10)]
        return full, test, labels

    if kind == "emnist_letters":
        full = RelabelDataset(
            datasets.EMNIST("data", split="letters", train=True, download=True, transform=emnist_tfm),
            subtract=1,
            offset=0,
        )
        test = RelabelDataset(
            datasets.EMNIST("data", split="letters", train=False, download=True, transform=emnist_tfm),
            subtract=1,
            offset=0,
        )
        labels = [chr(ord("A") + i) for i in range(26)]
        return full, test, labels

    # Combined 36-class model: digits 0-9 + uppercase letters A-Z.
    mnist_train = datasets.MNIST("data", train=True, download=True, transform=mnist_tfm)
    mnist_test = datasets.MNIST("data", train=False, download=True, transform=mnist_tfm)

    emnist_train = RelabelDataset(
        datasets.EMNIST("data", split="letters", train=True, download=True, transform=emnist_tfm),
        subtract=1,
        offset=10,
    )
    emnist_test = RelabelDataset(
        datasets.EMNIST("data", split="letters", train=False, download=True, transform=emnist_tfm),
        subtract=1,
        offset=10,
    )

    full = ConcatDataset([mnist_train, emnist_train])
    test = ConcatDataset([mnist_test, emnist_test])
    labels = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
    return full, test, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["mnist", "emnist_letters", "alphanumeric"],
        default="alphanumeric",
        help="mnist=digits only, emnist_letters=A-Z only, alphanumeric=0-9 + A-Z",
    )
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(42)
    full, test, labels = build_datasets(args.dataset)

    n_val = int(0.1 * len(full))
    train, val = random_split(
        full,
        [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=2)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset} | Classes: {len(labels)}")

    model = CharacterCNN(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    best = -1
    hist = []

    def eval_loader(loader, collect=False):
        model.eval()
        correct = total = 0
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p = model(x).argmax(1)
                correct += (p == y).sum().item()
                total += len(y)
                if collect:
                    ys.extend(y.cpu().numpy())
                    ps.extend(p.cpu().numpy())
        return correct / total, np.array(ys), np.array(ps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)

        val_acc, _, _ = eval_loader(val_loader)
        row = {
            "epoch": epoch,
            "train_loss": running / len(train),
            "val_acc": val_acc,
        }
        hist.append(row)
        print(row)

        if val_acc > best:
            best = val_acc
            torch.save(
                {
                    "state": model.state_dict(),
                    "labels": labels,
                    "dataset": args.dataset,
                },
                out / "best_model.pt",
            )

    ckpt = torch.load(out / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state"])
    test_acc, y_true, y_pred = eval_loader(test_loader, True)

    (out / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "classes": labels,
                "validation_best_accuracy": best,
                "test_accuracy": test_acc,
                "history": hist,
            },
            indent=2,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    size = 13 if len(labels) > 20 else 9
    fig = plt.figure(figsize=(size, size))
    plt.imshow(cm)
    plt.title(f"{args.dataset} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(f"Best validation accuracy: {best:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
