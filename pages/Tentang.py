import streamlit as st

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Tentang - FloodSeg",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM YANG DIPERBAIKI ---
st.markdown("""
    <style>
    /* Mengatur latar belakang utama aplikasi */
    .stApp {
        background-color: ;
    }
    
    /* Style untuk Kartu Putih (Konten) */
    .about-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: #333333; /* PENTING: Memaksa teks jadi gelap */
    }
    
    /* Memastikan heading di dalam card juga berwarna gelap */
    .about-card h1, .about-card h2, .about-card h3, .about-card h4, .about-card h5 {
        color: #2c3e50 !important;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .about-card p, .about-card li {
        color: #4a5568;
        line-height: 1.6;
    }

    /* Style untuk Kartu Tim (Gradient) */
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px 20px;
        border-radius: 15px;
        color: white !important; /* Memaksa teks tim jadi putih */
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .team-card h3 {
        color: white !important;
        margin-top: 10px;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
    }
    
    /* Badge Teknologi */
    .tech-badge {
        display: inline-block;
        background-color: #e2e8f0;
        color: #2d3748;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9em;
        font-weight: 600;
        border: 1px solid #cbd5e0;
    }
    
    /* Menghilangkan padding default Streamlit yang berlebihan di atas */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=80)
    st.title("ℹ️ Tentang")
    st.markdown("---")
    st.info("Informasi lengkap tentang FloodSeg dan teknologi yang digunakan.")

# --- HEADER ---
st.title("ℹ️ Tentang FloodSeg")
st.markdown("##### Sistem Deteksi Banjir Berbasis Deep Learning untuk Analisis Citra Satelit")
st.write("") # Spacer

# --- DESKRIPSI PROYEK ---
st.markdown("---")
st.header("🎯 Apa itu FloodSeg?")

st.markdown("""
<div class="about-card">
    <p>
        <strong>FloodSeg</strong> adalah aplikasi web berbasis <strong>Deep Learning</strong> yang dikembangkan 
        untuk mendeteksi dan menganalisis area yang terdampak banjir dari citra satelit. Aplikasi ini menggunakan 
        arsitektur <strong>U-Net</strong>, sebuah model Convolutional Neural Network (CNN) yang dirancang khusus 
        untuk tugas <strong>Image Segmentation</strong>.
    </p>
    <p>
        Dengan FloodSeg, pengguna dapat mengunggah citra satelit dan mendapatkan hasil segmentasi otomatis yang 
        memisahkan area perairan/banjir dari daratan. Hasil analisis disajikan dalam bentuk visualisasi interaktif 
        dengan metrik evaluasi yang lengkap.
    </p>
</div>
""", unsafe_allow_html=True)

# --- TUJUAN PROYEK ---
st.header("🎓 Tujuan Proyek")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="about-card" style="height: 100%;">
        <h3 style="border-bottom: 2px solid #667eea; padding-bottom: 10px; display:inline-block;">📚 Tujuan Akademik</h3>
        <ul>
            <li>Implementasi algoritma Deep Learning U-Net</li>
            <li>Penerapan teknik Image Processing</li>
            <li>Evaluasi performa model segmentasi</li>
            <li>Pengembangan aplikasi web interaktif</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="about-card" style="height: 100%;">
        <h3 style="border-bottom: 2px solid #667eea; padding-bottom: 10px; display:inline-block;">🌍 Tujuan Praktis</h3>
        <ul>
            <li>Membantu analisis dampak banjir</li>
            <li>Monitoring area perairan dari satelit</li>
            <li>Mendukung pengambilan keputusan</li>
            <li>Visualisasi data spasial yang mudah dipahami</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
# --- ARSITEKTUR U-NET ---
st.markdown("---")
st.header("🧠 Arsitektur U-Net")

st.markdown("""
<div class="about-card">
<h3 style="color: #2c3e50; margin-bottom: 15px;">Mengapa U-Net?</h3>
<p style="color: #4a5568; line-height: 1.6;">
<strong>U-Net</strong> adalah arsitektur CNN yang dikembangkan oleh Olaf Ronneberger et al. (2015) 
untuk <em>biomedical image segmentation</em>. Arsitektur ini memiliki karakteristik unik berbentuk "U" yang terdiri dari dua bagian utama:
</p>

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 25px; margin-bottom: 25px;">

<div style="flex: 1; min-width: 250px; background: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;">
<h4 style="color: #667eea; margin-top: 0; margin-bottom: 10px;">⬇️ Contracting Path (Encoder)</h4>
<p style="color: #4a5568; font-size: 0.95rem; margin: 0; line-height: 1.5;">
Berfungsi sebagai pengekstraksi fitur. Menggunakan <em>convolution layers</em> dan <em>max pooling</em> untuk menangkap konteks global dari gambar (downsampling).
</p>
</div>

<div style="flex: 1; min-width: 250px; background: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;">
<h4 style="color: #667eea; margin-top: 0; margin-bottom: 10px;">⬆️ Expanding Path (Decoder)</h4>
<p style="color: #4a5568; font-size: 0.95rem; margin: 0; line-height: 1.5;">
Berfungsi untuk lokalisasi presisi. Melakukan <em>upsampling</em> untuk mengembalikan resolusi gambar ke ukuran aslinya agar sesuai dengan input.
</p>
</div>

</div>

<div style="background-color: #eef2ff; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea;">
<p style="color: #4a5568; margin: 0; line-height: 1.6;">
<strong>🔗 Skip Connections:</strong> 
Fitur kunci yang menghubungkan layer encoder langsung ke decoder. Ini memungkinkan model mempertahankan informasi spasial detail yang biasanya hilang saat downsampling, menghasilkan segmentasi yang sangat akurat pada tepian objek.
</p>
</div>

</div>
""", unsafe_allow_html=True)

# Diagram sederhana (Menggunakan st.code agar rapi di semua tema)
st.subheader("📊 Alur Arsitektur")
st.code("""
INPUT CITRA (256×256×3)
    │
    ▼
┌───────────────────────┐
│ ENCODER (Downsample)  │ ➜ Ekstraksi Fitur (Conv + MaxPool)
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ BOTTLENECK            │ ➜ Representasi Fitur Terdalam
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│ DECODER (Upsample)    │ ➜ Rekonstruksi (UpConv + Concatenation)
└───────────────────────┘
    │
    ▼
OUTPUT MASK (256×256×1) ➜ Prediksi Piksel (Air vs Darat)
""", language="text")
# --- TEKNOLOGI ---
st.markdown("---")
st.header("💻 Teknologi yang Digunakan")

st.markdown("""
<div class="about-card">
<h3 style="color: #2c3e50; margin-bottom: 25px; text-align: center;">Stack Teknologi</h3>

<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 30px; text-align: center; margin-bottom: 20px;">

<div style="flex: 1; min-width: 120px; max-width: 200px;">
<div style="font-size: 3.5rem; margin-bottom: 10px;">🐍</div>
<h4 style="color: #2c3e50; margin: 0;">Python 3.x</h4>
<p style="color: #6c757d; font-size: 0.85rem; margin-top: 5px;">Bahasa Pemrograman</p>
</div>

<div style="flex: 1; min-width: 120px; max-width: 200px;">
<div style="font-size: 3.5rem; margin-bottom: 10px;">🧠</div>
<h4 style="color: #2c3e50; margin: 0;">TensorFlow</h4>
<p style="color: #6c757d; font-size: 0.85rem; margin-top: 5px;">Deep Learning</p>
</div>

<div style="flex: 1; min-width: 120px; max-width: 200px;">
<div style="font-size: 3.5rem; margin-bottom: 10px;">🎨</div>
<h4 style="color: #2c3e50; margin: 0;">Streamlit</h4>
<p style="color: #6c757d; font-size: 0.85rem; margin-top: 5px;">Web Framework</p>
</div>

</div>

<hr style="margin: 25px 0; border: 0; border-top: 1px solid #e1e4e8;">

<div style="text-align: center;">
<p style="color: #2c3e50; margin-bottom: 15px; font-weight: 600;">Libraries & Modules:</p>
<span class="tech-badge">TensorFlow 2.x</span>
<span class="tech-badge">NumPy</span>
<span class="tech-badge">OpenCV</span>
<span class="tech-badge">Pillow</span>
<span class="tech-badge">Streamlit</span>
<span class="tech-badge">Pandas</span>
<span class="tech-badge">Matplotlib</span>
</div>
</div>
""", unsafe_allow_html=True)
# --- FITUR UTAMA ---
st.markdown("---")
st.header("✨ Fitur Utama")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
<div class="about-card" style="height: 100%;">
<h4 style="color: #667eea; margin-bottom: 5px;">🎯 Segmentasi Otomatis</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 20px;">
Deteksi area air/banjir secara otomatis menggunakan model U-Net terlatih.
</p>
<h4 style="color: #667eea; margin-bottom: 5px;">📊 Visualisasi Interaktif</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 20px;">
Tampilan hasil dengan mask binary dan overlay transparan yang dapat disesuaikan.
</p>
<h4 style="color: #667eea; margin-bottom: 5px;">📈 Metrik Evaluasi</h4>
<p style="color: #4a5568; margin-top: 0;">
Perhitungan IoU, Dice Coefficient, dan Accuracy untuk validasi hasil.
</p>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="about-card" style="height: 100%;">
<h4 style="color: #667eea; margin-bottom: 5px;">⚙️ Pengaturan Fleksibel</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 20px;">
Threshold dan transparansi dapat disesuaikan sesuai kebutuhan analisis.
</p>
<h4 style="color: #667eea; margin-bottom: 5px;">💾 Download Hasil</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 20px;">
Export mask dan overlay dalam format PNG untuk dokumentasi.
</p>
<h4 style="color: #667eea; margin-bottom: 5px;">🚨 Status Banjir</h4>
<p style="color: #4a5568; margin-top: 0;">
Klasifikasi otomatis tingkat bahaya (Aman/Waspada/Bahaya).
</p>
</div>
""", unsafe_allow_html=True)


# --- REFERENSI ---
st.markdown("---")
st.header("📚 Referensi")

st.markdown("""
<div class="about-card">
    <ol style="padding-left: 20px;">
        <li style="margin-bottom: 10px;">
            <strong>U-Net: Convolutional Networks for Biomedical Image Segmentation</strong><br>
            Ronneberger, O., Fischer, P., & Brox, T. (2015)<br>
            <a href="https://arxiv.org/abs/1505.04597" target="_blank" style="color: #667eea;">Link Paper</a>
        </li>
        <li style="margin-bottom: 10px;">
            <strong>TensorFlow Documentation</strong><br>
            <a href="https://www.tensorflow.org/" target="_blank" style="color: #667eea;">Link Docs</a>
        </li>
        <li style="margin-bottom: 10px;">
            <strong>Streamlit Documentation</strong><br>
            <a href="https://docs.streamlit.io/" target="_blank" style="color: #667eea;">Link Docs</a>
        </li>
    </ol>
</div>
""", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666; font-size: 0.9rem;">
    <p><strong>FloodSeg v1.0</strong> © 2025</p>
    <p>Dikembangkan dengan ❤️ untuk Tugas Pengolahan Citra Digital</p>
</div>
""", unsafe_allow_html=True)