from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import streamlit as st
from model import CharacterCNN

st.set_page_config(page_title="Handwritten Character Recognition",page_icon="✍️")
st.title("✍️ Handwritten Character Recognition")
st.caption("CodeAlpha Machine Learning Task 3 — adaptive-pooling CNN on MNIST / EMNIST")

model_path=Path("artifacts/best_model.pt")
if not model_path.exists(): st.info("Train the model first with `python train.py`."); st.stop()
device=torch.device("cpu"); ckpt=torch.load(model_path,map_location=device); labels=ckpt["labels"]; model=CharacterCNN(len(labels)); model.load_state_dict(ckpt["state"]); model.eval()

uploaded=st.file_uploader("Upload a handwritten digit/character image",type=["png","jpg","jpeg"])
if uploaded:
    img=Image.open(uploaded).convert("L")
    img=ImageOps.autocontrast(img)
    arr=np.array(img)
    if arr.mean()>127: img=ImageOps.invert(img)
    img=ImageOps.fit(img,(28,28))
    st.image(img.resize((280,280)),caption="Preprocessed 28×28 input")
    x=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])(img).unsqueeze(0)
    with torch.no_grad(): probs=torch.softmax(model(x),1)[0].numpy()
    order=np.argsort(probs)[::-1][:5]
    st.subheader(f"Prediction: {labels[order[0]]}")
    for i in order: st.write(f"**{labels[i]}** — {probs[i]:.1%}")
