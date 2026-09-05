import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from features import extract_mfcc
from model import EmotionConvGRUAttention

RAVDESS = {"01":"neutral","02":"neutral","03":"happy","04":"sad","05":"angry","06":"fear","07":"disgust","08":"surprise"}
TESS = {"angry":"angry","disgust":"disgust","fear":"fear","happy":"happy","neutral":"neutral","sad":"sad","ps":"surprise","surprise":"surprise"}
EMODB = {"W":"angry","E":"disgust","A":"fear","F":"happy","T":"sad","N":"neutral"}


def scan(kind, root):
    rows = []
    for p in Path(root).rglob("*.wav"):
        if kind == "ravdess":
            parts = p.stem.split("-")
            if len(parts) >= 3 and parts[2] in RAVDESS:
                rows.append((str(p), RAVDESS[parts[2]]))
        elif kind == "tess":
            text = f"{p.parent.name}_{p.stem}".lower()
            for key, val in TESS.items():
                if key in text:
                    rows.append((str(p), val)); break
        else:
            if len(p.stem) >= 6 and p.stem[5].upper() in EMODB:
                rows.append((str(p), EMODB[p.stem[5].upper()]))
    return rows


class AudioDataset(Dataset):
    def __init__(self, rows, label_to_id, augment=False):
        self.rows, self.label_to_id, self.augment = rows, label_to_id, augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, label = self.rows[i]
        return torch.tensor(extract_mfcc(p, self.augment)), torch.tensor(self.label_to_id[label], dtype=torch.long)


def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; losses=[]
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(loss_fn(logits, y).item() * len(y))
            ys.extend(y.cpu().numpy()); ps.extend(logits.argmax(1).cpu().numpy())
    return sum(losses)/len(loader.dataset), accuracy_score(ys, ps), np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ravdess","tess","emodb"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    rows = scan(args.dataset, args.data_dir)
    if len(rows) < 20:
        raise RuntimeError(f"Only {len(rows)} labelled WAV files found. Check dataset path/layout.")

    labels = sorted({y for _, y in rows}); label_to_id = {x:i for i,x in enumerate(labels)}
    train_rows, test_rows = train_test_split(rows, test_size=.2, random_state=42, stratify=[y for _,y in rows])
    train_rows, val_rows = train_test_split(train_rows, test_size=.15, random_state=42, stratify=[y for _,y in train_rows])
    train_loader = DataLoader(AudioDataset(train_rows,label_to_id,True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_rows,label_to_id), batch_size=args.batch_size)
    test_loader = DataLoader(AudioDataset(test_rows,label_to_id), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EmotionConvGRUAttention(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    history=[]; best=-1; bad=0

    for epoch in range(1, args.epochs+1):
        model.train(); total=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); total += loss.item()*len(y)
        val_loss,val_acc,_,_=evaluate(model,val_loader,device)
        history.append({"epoch":epoch,"train_loss":total/len(train_loader.dataset),"val_loss":val_loss,"val_acc":val_acc})
        print(history[-1])
        if val_acc > best:
            best=val_acc; bad=0
            torch.save({"model_state":model.state_dict(),"labels":labels}, out/"best_model.pt")
        else:
            bad += 1
            if bad >= 8: break

    ckpt=torch.load(out/"best_model.pt",map_location=device); model.load_state_dict(ckpt["model_state"])
    _,acc,y_true,y_pred=evaluate(model,test_loader,device)
    report=classification_report(y_true,y_pred,target_names=labels,digits=4,zero_division=0)
    (out/"classification_report.txt").write_text(f"Test accuracy: {acc:.4f}\n\n{report}")
    (out/"history.json").write_text(json.dumps(history,indent=2))
    cm=confusion_matrix(y_true,y_pred)
    fig=plt.figure(figsize=(8,7)); plt.imshow(cm); plt.xticks(range(len(labels)),labels,rotation=45,ha="right"); plt.yticks(range(len(labels)),labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Speech Emotion Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)): plt.text(j,i,cm[i,j],ha="center",va="center")
    plt.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=180); plt.close(fig)
    print(report)

if __name__ == "__main__": main()
