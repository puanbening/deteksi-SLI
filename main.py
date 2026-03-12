import streamlit as st
import numpy as np
import librosa
import lightgbm as lgb
import joblib
import json
import tensorflow as tf
import tensorflow_hub as tf_hub
import tempfile
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi SLI",
    page_icon="🎙️",
    layout="centered"
)

# ─── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = lgb.Booster(model_file="model_lgb.txt")
    scaler = joblib.load("scaler.pkl")
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]
    yamnet = tf_hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)
    return model, scaler, threshold, yamnet

model_lgb, scaler, optimal_threshold, yamnet_layer = load_resources()

# ─── Audio Processing Functions ────────────────────────────────────────────────
def load_audio_waveform(audio_path, sr=16000):
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(waveform) == 0:
        return None
    waveform, _ = librosa.effects.trim(waveform, top_db=25)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-9)
    return waveform

def chunk_waveform(waveform, sr=16000, chunk_duration=0.96):
    chunk_len = int(sr * chunk_duration)
    if len(waveform) < chunk_len:
        waveform = np.pad(waveform, (0, chunk_len - len(waveform)))
    return [
        waveform[i:i + chunk_len]
        for i in range(0, len(waveform) - chunk_len + 1, chunk_len)
    ]

def extract_mfcc_features(chunk, sr=16000, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

def extract_hybrid_features(waveform, sr=16000):
    chunks = chunk_waveform(waveform, sr)
    if not chunks:
        return None
    hybrid_feats = []
    for chunk in chunks:
        chunk_tf = tf.convert_to_tensor(chunk, dtype=tf.float32)
        mfcc_feat = extract_mfcc_features(chunk, sr)
        _, embeddings, _ = yamnet_layer(chunk_tf)
        yamnet_feat = tf.reduce_mean(embeddings, axis=0).numpy()
        hybrid_feats.append(np.concatenate([mfcc_feat, yamnet_feat]))
    return np.mean(hybrid_feats, axis=0)

def is_noisy(waveform):
    energy = np.mean(np.square(waveform))
    return energy < 1e-4

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("Deteksi Dini Specific Language Impairment (SLI)")
st.markdown(
    "Upload file audio untuk mendeteksi kemungkinan "
    "adanya gangguan *Specific Language Impairment* (SLI)."
)
st.divider()

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav","mp3","m4a","ogg","flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)
    

    with st.spinner("Memproses audio..."):
        # ambil ekstensi file asli
        ext = os.path.splitext(uploaded_file.name)[1]

        # simpan file sementara dengan ekstensi yang sama
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            waveform = load_audio_waveform(tmp_path)

            if waveform is None or len(waveform) == 0:
                st.error("❌ File audio tidak valid atau terlalu pendek.")
            else:
                feat = extract_hybrid_features(waveform)
                feat_scaled = scaler.transform(feat.reshape(1, -1))
                prob = model_lgb.predict(feat_scaled)[0]

                st.divider()
                st.subheader("📊 Hasil Deteksi")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Skor Prediksi", f"{prob:.4f}")
                with col2:
                    st.metric("Threshold", f"{optimal_threshold:.4f}")

                # Tentukan hasil
                if is_noisy(waveform):
                    st.warning("⚠️ **Audio terdeteksi mengandung noise berlebihan.** Hasil deteksi mungkin tidak akurat.")
                elif prob >= optimal_threshold:
                    st.success("✅ **Hasil: Sehat (Healthy)**")
                    st.markdown("Audio anak tidak menunjukkan indikasi gangguan SLI.")
                else:
                    st.error("🔴 **Hasil: Terdeteksi SLI**")
                    st.markdown(
                        "Audio anak menunjukkan indikasi gangguan *Specific Language Impairment*. "
                        "Disarankan untuk berkonsultasi dengan terapis wicara."
                    )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
        finally:
            os.remove(tmp_path)

st.divider()
st.caption(
    "⚠️ Aplikasi ini merupakan alat bantu skrining awal dan **bukan pengganti diagnosis klinis**. "
    "Selalu konsultasikan hasil dengan tenaga medis profesional."
)
