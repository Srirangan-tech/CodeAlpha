from pathlib import Path
import numpy as np
import torch
from features import extract_mfcc
from model import EmotionConvGRUAttention


def predict(path, artifact_dir="artifacts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(Path(artifact_dir)/"best_model.pt", map_location=device)
    labels = ckpt["labels"]
    model = EmotionConvGRUAttention(len(labels)).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    x = torch.tensor(extract_mfcc(path)).unsqueeze(0).to(device)
    with torch.no_grad(): probs = torch.softmax(model(x),1)[0].cpu().numpy()
    order=np.argsort(probs)[::-1]
    return [(labels[i],float(probs[i])) for i in order]
