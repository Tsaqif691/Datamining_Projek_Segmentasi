import streamlit as st

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Tutorial - FloodSeg",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .tutorial-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        font-size: 1.2rem;
        margin-right: 15px;
    }
    .tip-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #155724;
    }
    .danger-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #721c24;
    }
    .code-box {
        background: #2d2d2d;
        color: #f8f8f2;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 15px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
        line-height: 1.6;
    }
    .tutorial-card {
        color: #333;
    }
    .tip-box {
        color: #004085;
    }
    .warning-box {
        color: #856404;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water.png", width=80)
    st.title("📚 Tutorial")
    
    st.markdown("---")
    st.markdown("### 📑 Daftar Isi")
    st.markdown("""
    - [Pengenalan](#pengenalan)
    - [Persiapan Data](#persiapan-data)
    - [Cara Upload](#cara-upload-gambar)
    - [Interpretasi Hasil](#interpretasi-hasil)
    - [Tips & Trik](#tips-trik)
    - [FAQ](#faq)
    """)

# --- HEADER ---
st.title("📚 Tutorial Penggunaan FloodSeg")
st.markdown("Panduan lengkap menggunakan sistem deteksi banjir berbasis AI")

# --- PENGENALAN ---
st.markdown("---")
st.markdown("## 🎯 Pengenalan")

st.markdown("""
<div class="tutorial-card">
    <p style="font-size: 1.1rem; line-height: 1.8;">
        Selamat datang di <strong>FloodSeg Tutorial</strong>! Panduan ini akan membantu Anda memahami 
        cara menggunakan aplikasi untuk menganalisis citra satelit dan mendeteksi area yang terdampak banjir.
    </p>
    <p style="font-size: 1.1rem; line-height: 1.8;">
        FloodSeg menggunakan teknologi <strong>Deep Learning U-Net</strong> untuk melakukan segmentasi 
        semantik pada citra, memisahkan area perairan/banjir dari daratan dengan akurasi tinggi.
    </p>
</div>
""", unsafe_allow_html=True)

# --- PERSIAPAN DATA ---
st.markdown("---")
st.markdown("## 📁 Persiapan Data")

st.markdown("""
<div class="tutorial-card">
    <h3><span class="step-number">1</span>Persyaratan Gambar Input</h3>
    <p style="margin-left: 55px; line-height: 1.8;">
        Untuk mendapatkan hasil terbaik, pastikan gambar yang akan diupload memenuhi kriteria berikut:
    </p>
    <ul style="margin-left: 55px; line-height: 2;">
        <li><strong>Format:</strong> JPG, PNG, atau JPEG</li>
        <li><strong>Ukuran:</strong> Maksimal 10 MB</li>
        <li><strong>Resolusi:</strong> Minimal 256x256 pixel (akan di-resize otomatis)</li>
        <li><strong>Konten:</strong> Citra satelit atau aerial view yang menampakkan area perairan</li>
        <li><strong>Kualitas:</strong> Citra dengan pencahayaan baik dan kontras yang jelas</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Contoh gambar yang baik vs buruk
st.markdown("### ✅ Contoh Gambar yang Baik vs ❌ Buruk")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="success-box">
        <h4>✅ Gambar yang Baik:</h4>
        <ul>
            <li>Kontras jelas antara air dan daratan</li>
            <li>Pencahayaan merata</li>
            <li>Tidak ada kabut atau awan tebal</li>
            <li>Resolusi tinggi dan fokus tajam</li>
            <li>Area perairan terlihat jelas</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
        <h4>❌ Gambar yang Kurang Baik:</h4>
        <ul>
            <li>Blur atau tidak fokus</li>
            <li>Tertutup awan/kabut tebal</li>
            <li>Pencahayaan terlalu gelap/terang</li>
            <li>Resolusi rendah (pixelated)</li>
            <li>Tidak menampakkan area perairan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")
st.markdown("## 📤 Cara Upload Gambar")

# Pastikan tag <div> menempel mentok ke kiri (tidak ada spasi di depannya)
st.markdown("""
<div class="tutorial-card">
<h3><span class="step-number">2</span>Panduan Langkah Upload</h3>
<div style="margin-top: 25px; margin-left: 10px;">
<div style="display: flex; margin-bottom: 20px;">
<div style="margin-right: 15px; font-size: 1.5rem;">🎯</div>
<div>
<h4 style="color: #2c3e50; margin: 0 0 5px 0;">Langkah 1: Buka Menu Segmentasi</h4>
<p style="color: #4a5568; margin: 0; font-size: 0.95rem;">
Lihat sidebar di sebelah kiri, lalu klik menu <strong>"🎯 Segmentasi"</strong>.
</p>
</div>
</div>
<div style="display: flex; margin-bottom: 20px;">
<div style="margin-right: 15px; font-size: 1.5rem;">📂</div>
<div>
<h4 style="color: #2c3e50; margin: 0 0 5px 0;">Langkah 2: Upload File</h4>
<p style="color: #4a5568; margin: 0; font-size: 0.95rem;">
Klik tombol <strong>"Browse files"</strong> atau <em>drag & drop</em> gambar satelit Anda ke area kotak upload.
</p>
</div>
</div>
<div style="display: flex; margin-bottom: 20px;">
<div style="margin-right: 15px; font-size: 1.5rem;">⏳</div>
<div>
<h4 style="color: #2c3e50; margin: 0 0 5px 0;">Langkah 3: Tunggu Proses</h4>
<p style="color: #4a5568; margin: 0; font-size: 0.95rem;">
Sistem akan memproses gambar otomatis (2-5 detik). Jangan tutup halaman saat loading.
</p>
</div>
</div>
<div style="display: flex;">
<div style="margin-right: 15px; font-size: 1.5rem;">✨</div>
<div>
<h4 style="color: #2c3e50; margin: 0 0 5px 0;">Langkah 4: Hasil Selesai</h4>
<p style="color: #4a5568; margin: 0; font-size: 0.95rem;">
Hasil segmentasi (Mask & Overlay) akan muncul di layar dan siap diunduh.
</p>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")
st.markdown("## ⚙️ Pengaturan Model")

st.markdown("""
<div class="tutorial-card">
<h3><span class="step-number">3</span>Menyesuaikan Parameter</h3>
<div style="margin-left: 10px; margin-top: 20px;">
<h4 style="color: #2c3e50; margin-bottom: 10px;">📊 Sensitivity Threshold</h4>
<p style="color: #4a5568; line-height: 1.6; font-size: 0.95rem;">
Slider ini mengontrol <strong>sensitivitas deteksi</strong> model. Nilai berkisar dari 0.0 hingga 1.0.
</p>
<ul style="color: #4a5568; line-height: 1.8; margin-bottom: 25px;">
<li><strong>Nilai Rendah (0.3 - 0.4):</strong> Lebih sensitif, deteksi lebih banyak area sebagai air.</li>
<li><strong>Nilai Sedang (0.5):</strong> Balanced, direkomendasikan untuk kebanyakan kasus.</li>
<li><strong>Nilai Tinggi (0.6 - 0.7):</strong> Lebih konservatif, hanya mendeteksi area dengan keyakinan tinggi.</li>
</ul>
<h4 style="color: #2c3e50; margin-bottom: 10px;">🎨 Transparansi Overlay</h4>
<p style="color: #4a5568; line-height: 1.6; font-size: 0.95rem;">
Mengatur tingkat transparansi warna cyan pada hasil overlay.
</p>
<ul style="color: #4a5568; line-height: 1.8;">
<li><strong>0.0 - 0.3:</strong> Hampir transparan (lebih terlihat gambar asli).</li>
<li><strong>0.4 - 0.6:</strong> Visibility seimbang (Balanced).</li>
<li><strong>0.7 - 1.0:</strong> Warna lebih solid (overlay lebih dominan).</li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)

# Contoh interaktif
st.markdown("### 🎮 Coba Atur Parameter")

col1, col2 = st.columns(2)

with col1:
    demo_threshold = st.slider(
        "Demo: Sensitivity Threshold",
        0.0, 1.0, 0.5, 0.05,
        help="Coba geser untuk lihat efeknya"
    )
    st.info(f"Threshold saat ini: **{demo_threshold:.2f}**")
    
    if demo_threshold < 0.4:
        st.warning("⚠️ Sensitivity tinggi - Dapat mendeteksi bayangan sebagai air")
    elif demo_threshold > 0.6:
        st.warning("⚠️ Sensitivity rendah - Mungkin melewatkan area air yang tipis")
    else:
        st.success("✅ Nilai optimal untuk kebanyakan kasus")

with col2:
    demo_alpha = st.slider(
        "Demo: Transparansi Overlay",
        0.0, 1.0, 0.4, 0.1,
        help="Coba geser untuk lihat efeknya"
    )
    st.info(f"Alpha saat ini: **{demo_alpha:.2f}**")
    
    if demo_alpha < 0.3:
        st.info("ℹ️ Overlay hampir tidak terlihat")
    elif demo_alpha > 0.7:
        st.info("ℹ️ Overlay sangat dominan")
    else:
        st.success("✅ Visualisasi seimbang")
# --- INTERPRETASI HASIL ---
st.markdown("---")
st.markdown("## 📊 Interpretasi Hasil")

st.markdown("""
<div class="tutorial-card">
<h3><span class="step-number">4</span>Memahami Output Model</h3>
<div style="margin-left: 10px; margin-top: 20px;">
<h4 style="color: #2c3e50; margin-bottom: 5px;">1. Citra Asli</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 15px;">
Gambar input yang telah di-resize otomatis ke resolusi 256x256 pixel.
</p>
<h4 style="color: #2c3e50; margin-bottom: 5px;">2. Hasil Masking</h4>
<p style="color: #4a5568; margin-top: 0;">
Binary mask (hitam-putih) dimana:
</p>
<ul style="color: #4a5568; margin-bottom: 15px;">
<li><strong>Putih (255):</strong> Area yang terdeteksi sebagai air/banjir.</li>
<li><strong>Hitam (0):</strong> Area yang terdeteksi sebagai daratan.</li>
</ul>
<h4 style="color: #2c3e50; margin-bottom: 5px;">3. Visualisasi Overlay</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 15px;">
Kombinasi gambar asli dengan mask yang diberi warna <strong>cyan transparan</strong> pada area air.
</p>
<h4 style="color: #2c3e50; margin-bottom: 5px;">4. Analisis Dampak Banjir</h4>
<p style="color: #4a5568; margin-top: 0;">
Menampilkan informasi statistik:
</p>
<ul style="color: #4a5568; margin-bottom: 0;">
<li><strong>Persentase Area Terdampak:</strong> Rasio pixel air terhadap total pixel.</li>
<li><strong>Status Level:</strong> Indikator Aman, Waspada, atau Bahaya.</li>
<li><strong>Deskripsi:</strong> Interpretasi kondisi banjir.</li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)

# Status Level Explanation
st.markdown("### 🚦 Kategori Status Banjir")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="success-box">
        <h4>🟢 Status: AMAN</h4>
        <p><strong>Persentase:</strong> < 15%</p>
        <p><strong>Arti:</strong> Area perairan dalam batas normal. Tidak ada indikasi banjir signifikan.</p>
        <p><strong>Tindakan:</strong> Monitoring rutin</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
        <h4>🟡 Status: WASPADA</h4>
        <p><strong>Persentase:</strong> 15% - 30%</p>
        <p><strong>Arti:</strong> Area terdampak cukup signifikan. Perlu pemantauan lebih ketat.</p>
        <p><strong>Tindakan:</strong> Siaga & monitoring aktif</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="danger-box">
        <h4>🔴 Status: BAHAYA</h4>
        <p><strong>Persentase:</strong> > 30%</p>
        <p><strong>Arti:</strong> Area terdampak sangat luas. Situasi darurat.</p>
        <p><strong>Tindakan:</strong> Evakuasi & mitigasi segera</p>
    </div>
    """, unsafe_allow_html=True)
# --- METRIK EVALUASI ---
st.markdown("---")
st.markdown("## 📈 Memahami Metrik Evaluasi")

st.markdown("""
<div class="tutorial-card">
<h3><span class="step-number">5</span>Metrik Performa Model</h3>
<div style="margin-left: 10px; margin-top: 20px;">
<h4 style="color: #2c3e50; margin-bottom: 5px;">📊 IoU (Intersection over Union)</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 10px;">
Mengukur seberapa besar <strong>overlap</strong> antara prediksi model dengan ground truth.
</p>
<div class="code-box" style="color: #d63384; background-color: #f8f9fa;">
IoU = Area of Overlap / Area of Union<br>
Range: 0.0 - 1.0 (semakin tinggi semakin baik)<br>
IoU > 0.9 = Excellent
</div>
<h4 style="color: #2c3e50; margin-top: 20px; margin-bottom: 5px;">🎯 Dice Coefficient</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 10px;">
Mirip dengan IoU, namun lebih sensitif terhadap <strong>ukuran objek</strong>.
</p>
<div class="code-box" style="color: #d63384; background-color: #f8f9fa;">
Dice = 2 × Area of Overlap / (Area1 + Area2)<br>
Range: 0.0 - 1.0<br>
Dice > 0.9 = Very Good
</div>
<h4 style="color: #2c3e50; margin-top: 20px; margin-bottom: 5px;">✅ Accuracy</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 10px;">
Persentase <strong>pixel yang diklasifikasikan dengan benar</strong>.
</p>
<div class="code-box" style="color: #d63384; background-color: #f8f9fa;">
Accuracy = Correct Pixels / Total Pixels<br>
Range: 0.0 - 1.0<br>
Accuracy > 0.95 = Highly Accurate
</div>
</div>
</div>
""", unsafe_allow_html=True)

# --- DOWNLOAD HASIL ---
st.markdown("---")
st.markdown("## 💾 Download Hasil")

st.markdown("""
<div class="tutorial-card">
<h3><span class="step-number">6</span>Export Hasil Segmentasi</h3>
<div style="margin-left: 10px; margin-top: 20px;">
<p style="color: #4a5568; line-height: 1.6;">
FloodSeg menyediakan <strong>2 opsi download</strong> untuk menyimpan hasil analisis Anda:
</p>
<h4 style="color: #2c3e50; margin-top: 20px; margin-bottom: 5px;">1️⃣ Download Mask</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 10px;">
File PNG berisi <strong>binary mask</strong> (hitam-putih) yang berguna untuk:
</p>
<ul style="color: #4a5568; margin-bottom: 15px;">
<li>Analisis lebih lanjut di software GIS (QGIS/ArcGIS).</li>
<li>Perhitungan area banjir dengan presisi tinggi.</li>
<li>Data validasi (Ground Truth) untuk training model lain.</li>
<li>Dokumentasi teknis hasil segmentasi.</li>
</ul>
<h4 style="color: #2c3e50; margin-top: 20px; margin-bottom: 5px;">2️⃣ Download Overlay</h4>
<p style="color: #4a5568; margin-top: 0; margin-bottom: 10px;">
File PNG berisi <strong>visualisasi overlay</strong> (gambar asli + warna biru) untuk:
</p>
<ul style="color: #4a5568; margin-bottom: 20px;">
<li>Presentasi visual ke klien atau publik.</li>
<li>Laporan dampak banjir yang mudah dipahami.</li>
<li>Sharing cepat melalui media sosial/pesan.</li>
</ul>
<div class="tip-box" style="color: #0c5460;">
<strong>💡 Tips:</strong> Kedua file otomatis diberi nama dengan tanggal & waktu (timestamp) agar file Anda tidak tertukar.
</div>
</div>
</div>
""", unsafe_allow_html=True)

# --- TIPS & TRIK ---
st.markdown("---")
st.markdown("## 💡 Tips & Trik")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tip-box">
        <h4>🎯 Untuk Hasil Terbaik:</h4>
        <ul>
            <li>Gunakan citra dengan resolusi minimal 512x512</li>
            <li>Pastikan kontras air-daratan jelas</li>
            <li>Hindari citra dengan awan tebal</li>
            <li>Upload pada siang hari untuk pencahayaan optimal</li>
            <li>Mulai dengan threshold default (0.5)</li>
        </ul>
    </div>
    
    <div class="success-box">
        <h4>✅ Best Practices:</h4>
        <ul>
            <li>Bandingkan hasil dengan citra sebelum banjir</li>
            <li>Simpan mask untuk analisis time-series</li>
            <li>Dokumentasikan pengaturan parameter</li>
            <li>Validasi hasil dengan ground truth jika ada</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Troubleshooting:</h4>
        <ul>
            <li><strong>Hasil terlalu banyak noise:</strong> Naikkan threshold</li>
            <li><strong>Area air tidak terdeteksi:</strong> Turunkan threshold</li>
            <li><strong>Gambar blur:</strong> Gunakan citra dengan kualitas lebih baik</li>
            <li><strong>Proses lambat:</strong> Kompres ukuran file gambar</li>
        </ul>
    </div>
    
    <div class="tip-box">
        <h4>🚀 Optimisasi Performa:</h4>
        <ul>
            <li>Kompres gambar sebelum upload</li>
            <li>Gunakan format JPG untuk file besar</li>
            <li>Crop area yang tidak relevan</li>
            <li>Batch processing untuk multiple images</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- FAQ ---
st.markdown("---")
st.markdown("## ❓ FAQ (Frequently Asked Questions)")

with st.expander("❓ Berapa lama waktu yang dibutuhkan untuk analisis?"):
    st.markdown("""
    Proses analisis biasanya memakan waktu **2-5 detik** tergantung:
    - Ukuran file gambar
    - Spesifikasi komputer/server
    - Beban sistem saat itu
    
    Model sudah di-cache, jadi loading pertama mungkin lebih lama (~10 detik).
    """)

with st.expander("❓ Apakah bisa menggunakan citra selain satelit?"):
    st.markdown("""
    **Ya**, model dapat bekerja dengan berbagai jenis citra aerial view:
    - Citra satelit (Landsat, Sentinel, dll)
    - Drone footage
    - Aerial photography
    - Google Earth screenshots
    
    Namun, hasil terbaik didapat dari citra satelit multispektral.
    """)

with st.expander("❓ Apakah data gambar yang diupload disimpan?"):
    st.markdown("""
    **Tidak**. Aplikasi ini:
    - ✅ Memproses gambar secara real-time di memory
    - ✅ Tidak menyimpan gambar ke server
    - ✅ Tidak mengumpulkan data pribadi
    - ✅ Session berakhir saat browser ditutup
    
    Data Anda **100% aman dan private**.
    """)

with st.expander("❓ Bagaimana cara meningkatkan akurasi deteksi?"):
    st.markdown("""
    Tips untuk akurasi lebih baik:
    1. **Gunakan citra berkualitas tinggi** (resolusi min 512x512)
    2. **Pilih waktu yang tepat** (hindari saat hujan/kabut)
    3. **Sesuaikan threshold** sesuai kondisi citra
    4. **Bandingkan dengan citra sebelumnya** untuk validasi
    5. **Gunakan citra multi-temporal** untuk deteksi perubahan
    """)

with st.expander("❓ Apakah bisa mendeteksi jenis air lainnya (sungai, danau, laut)?"):
    st.markdown("""
    **Ya!** Model U-Net dilatih untuk mendeteksi **semua jenis badan air**:
    - 🌊 Laut & samudra
    - 🏞️ Danau & waduk
    - 🏞️ Sungai & kanal
    - 💧 Genangan banjir
    - 🏊 Kolam renang (jika cukup besar)
    
    Model **tidak membedakan** jenis air, hanya membedakan air vs daratan.
    """)

with st.expander("❓ Bisakah hasil diintegrasikan dengan software GIS?"):
    st.markdown("""
    **Ya**, mask PNG yang didownload dapat digunakan di:
    - **QGIS** - Import sebagai raster layer
    - **ArcGIS** - Georeferencing & analysis
    - **Google Earth Engine** - Cloud processing
    - **Python (GDAL/Rasterio)** - Custom processing
    
    Untuk georeferencing, Anda perlu menambahkan metadata koordinat secara manual.
    """)

# --- TOMBOL AKSI ---
st.markdown("---")
st.markdown("## 🚀 Siap Mencoba?")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="color: #667eea;">Sudah Paham Cara Penggunaan?</h3>
        <p style="font-size: 1.1rem; margin: 20px 0; color: #333;">
            Mulai analisis citra satelit Anda sekarang!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 Mulai Segmentasi Sekarang", use_container_width=True):
        st.switch_page("pages/1_🎯_Segmentasi.py")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p>Butuh bantuan lebih lanjut? Hubungi kami di halaman <strong>Tentang</strong></p>
    <p><strong>FloodSeg</strong> - Making Flood Detection Accessible</p>
</div>
""", unsafe_allow_html=True)
