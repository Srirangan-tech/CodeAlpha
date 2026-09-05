import tempfile
import pandas as pd
import streamlit as st
from predict import predict

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Emotion Recognition from Speech")
st.caption("CodeAlpha Machine Learning Task 2 — MFCC + CNN + BiGRU + attention")

uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
recorded = st.audio_input("Or record a voice sample") if hasattr(st, "audio_input") else None
audio = recorded or uploaded

if audio:
    st.audio(audio)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio.getbuffer()); path=f.name
    try:
        result=predict(path)
        st.subheader(f"Prediction: {result[0][0].title()}")
        st.metric("Confidence",f"{result[0][1]:.1%}")
        df=pd.DataFrame(result,columns=["Emotion","Probability"]).set_index("Emotion")
        st.bar_chart(df)
    except FileNotFoundError:
        st.error("Train the model first using train.py.")
