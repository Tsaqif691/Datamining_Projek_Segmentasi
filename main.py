import streamlit as st

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="FloodSeg - Water Segmentation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .hero-section {
        padding: 60px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 30px;
    }
    .feature-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=100)
    st.title("🌊 FloodSeg")
    st.markdown("### Navigasi")
    st.info("Gunakan menu di atas untuk navigasi antar halaman")
    
    st.markdown("---")
    st.markdown("### 📌 Quick Info")
    st.write("**Model:** U-Net Deep Learning")
    st.write("**Akurasi:** ~95%")
    st.write("**Input:** 256x256 px")
    
    st.markdown("---")
    st.caption("© 2025 FloodSeg Team")

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🌊 FloodSeg</h1>
        <p class="hero-subtitle">Sistem Deteksi Area Banjir Berbasis Deep Learning</p>
        <p style="font-size: 1.1rem;">Analisis citra satelit untuk mengidentifikasi area terdampak banjir dengan teknologi U-Net</p>
    </div>
    """, unsafe_allow_html=True)

# --- FEATURES SECTION ---
st.markdown("## ✨ Fitur Unggulan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card" style="color: #333;">
        <h3 style="color: #667eea;">🎯 Akurasi Tinggi</h3>
        <p>Model U-Net dengan akurasi hingga 95% dalam mendeteksi area perairan dan banjir dari citra satelit.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="color: #333;">
        <h3 style="color: #667eea;">⚡ Proses Cepat</h3>
        <p>Analisis real-time dengan hasil segmentasi yang dapat dilihat dalam hitungan detik.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="color: #333;">
        <h3 style="color: #667eea;">📊 Visualisasi Interaktif</h3>
        <p>Hasil segmentasi ditampilkan dengan overlay interaktif dan metrik analisis lengkap.</p>
    </div>
    """, unsafe_allow_html=True)

# --- HOW IT WORKS ---
st.markdown("---")
st.markdown("## 🔬 Cara Kerja")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 3rem;">📤</div>
        <h4>1. Upload</h4>
        <p>Upload gambar satelit</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 3rem;">🤖</div>
        <h4>2. Analisis AI</h4>
        <p>Model U-Net memproses</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 3rem;">🎨</div>
        <h4>3. Segmentasi</h4>
        <p>Deteksi area air/banjir</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 3rem;">📈</div>
        <h4>4. Hasil</h4>
        <p>Visualisasi & metrik</p>
    </div>
    """, unsafe_allow_html=True)

# --- CTA SECTION ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="color: #667eea; margin-bottom: 20px; ">Siap Mencoba?</h2>
        <p style="font-size: 1.1rem; margin-bottom: 30px; color: #333;">Mulai analisis citra satelit Anda sekarang!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Mulai Analisis Segmentasi", use_container_width=True):
        st.switch_page("pages/Segmentasi.py")

# --- STATISTICS ---
st.markdown("---")
st.markdown("## 📊 Statistik Penggunaan")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Analisis", value="1,234", delta="↑ 23%")

with col2:
    st.metric(label="Akurasi Rata-rata", value="94.5%", delta="↑ 1.2%")

with col3:
    st.metric(label="Waktu Proses", value="2.3s", delta="↓ 0.5s")

with col4:
    st.metric(label="Pengguna Aktif", value="156", delta="↑ 12")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>FloodSeg</strong> - Tugas Pengolahan Citra Digital</p>
    <p>Dikembangkan dengan ❤️ menggunakan Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)