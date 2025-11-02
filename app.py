import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================
# 🌤️ Konfigurasi halaman Streamlit
# =====================================
st.set_page_config(page_title="Prediksi Kualitas Udara (NO₂)", layout="centered")
st.title("🌤️ Prediksi Kualitas Udara (NO₂) - Model Random Forest")
st.markdown("Masukkan data NO₂ beberapa hari sebelumnya untuk memprediksi konsentrasi hari esok.")

# =====================================
# 🧩 Fungsi aman untuk load model & scaler
# =====================================
def safe_load(file_path):
    """Load file pickle secara aman"""
    if not os.path.exists(file_path):
        st.error(f"❌ File `{file_path}` tidak ditemukan di direktori ini.")
        st.stop()
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Gagal memuat `{file_path}`: {e}")
        st.stop()

# =====================================
# 📦 Load model dan scaler
# =====================================
model = safe_load("model_rf_day5.pkl")
scaler_X = safe_load("scalerX_day5.pkl")
scaler_y = safe_load("scalerY_day5.pkl")

# =====================================
# 📥 Input manual
# =====================================
st.subheader("🧮 Masukkan Nilai NO₂ Sebelumnya (t-1 s/d t-5)")
t1 = st.number_input("t-1 (hari sebelumnya)", value=0.00005, format="%.8f")
t2 = st.number_input("t-2", value=0.00007, format="%.8f")
t3 = st.number_input("t-3", value=0.00009, format="%.8f")
t4 = st.number_input("t-4", value=0.00006, format="%.8f")
t5 = st.number_input("t-5", value=0.00008, format="%.8f")

# =====================================
# 🔮 Tombol Prediksi
# =====================================
if st.button("🔮 Prediksi Hari Esok"):
    # Buat DataFrame input
    input_data = pd.DataFrame([[t1, t2, t3, t4, t5]], columns=["t1", "t2", "t3", "t4", "t5"])

    # Tambahkan kolom dummy agar cocok dengan scaler_X yang dilatih
    for missing_col in ['feature_index', 'NO2']:
        input_data[missing_col] = 0

    # Urutkan kolom sesuai urutan saat scaler fit
    input_data = input_data[[col for col in scaler_X.feature_names_in_ if col in input_data.columns]]

    # === Normalisasi fitur
    input_scaled = scaler_X.transform(input_data)

    # === Prediksi dalam skala normalisasi
    pred_scaled = model.predict(input_scaled).reshape(-1, 1)

    # === Denormalisasi ke skala asli (pakai scaler_y)
    prediction = scaler_y.inverse_transform(pred_scaled)[0, 0]

    # =====================================
    # 🌍 Konversi ke satuan µg/m³ (WHO)
    # =====================================
    pred_ugm3 = prediction * 2_000_000  # konversi ke µg/m³ (perkiraan)
    
    if pred_ugm3 <= 40:
        kategori = "Sangat Baik ✅"
        warna = "🟢"
    elif pred_ugm3 <= 100:
        kategori = "Baik 😊"
        warna = "🟢"
    elif pred_ugm3 <= 200:
        kategori = "Sedang ⚠️"
        warna = "🟡"
    elif pred_ugm3 <= 400:
        kategori = "Buruk 🚨"
        warna = "🟠"
    else:
        kategori = "Sangat Buruk ☠️"
        warna = "🔴"

    # =====================================
    # 📊 Tampilkan hasil
    # =====================================
    st.success(f"🌱 Prediksi Konsentrasi NO₂ Hari Esok: **{prediction:.8f}** (≈ {pred_ugm3:.2f} µg/m³)")
    st.info(f"**Kategori Udara (WHO): {kategori} {warna}**")
    st.caption("📘 Berdasarkan WHO Air Quality Guidelines 2021 (NO₂ ≤ 200 µg/m³ untuk 1 jam).")

    # =====================================
    # 📈 Visualisasi mini
    # =====================================
    st.subheader("📈 Grafik Prediksi")
    df_plot = pd.DataFrame({
        "Hari": ["t-5", "t-4", "t-3", "t-2", "t-1", "Prediksi (t)"],
        "NO₂ (Relatif)": [t5, t4, t3, t2, t1, prediction]
    })
    st.line_chart(df_plot.set_index("Hari"))

# =====================================
# 🧾 Catatan Footer
# =====================================
st.markdown("---")
st.caption("🚀 Dibuat oleh Yudha Caesar Maulana • Model Random Forest Regressor dengan normalisasi MinMax berdasarkan data historis NO₂ harian.")
