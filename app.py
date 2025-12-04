# app.py - SISTEM REKOMENDASI POLA MAKAN SEHAT
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ====================== KONFIGURASI HALAMAN ======================
st.set_page_config(
    page_title="Sistem Rekomendasi Pola Makan Sehat",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CSS CUSTOM ======================
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3CB371;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3CB371;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    .recommendation-card {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 5px solid #4CAF50;
    }
    .health-good { color: #2E8B57; font-weight: bold; }
    .health-moderate { color: #FFA500; font-weight: bold; }
    .health-poor { color: #DC143C; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ====================== JUDUL APLIKASI ======================
st.markdown('<h1 class="main-title">🍎 SISTEM REKOMENDASI POLA MAKAN SEHAT</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Berdasarkan Klasterisasi Menu Fast Food dengan Metode K-Means Clustering</p>', unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("## ⚙️ KONFIGURASI")
    
    st.markdown("### 👤 PROFIL PENGGUNA")
    usia = st.number_input("Usia", min_value=15, max_value=70, value=25)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    berat_badan = st.number_input("Berat Badan (kg)", min_value=40, max_value=150, value=65)
    tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=140, max_value=200, value=170)
    
    st.markdown("### 🎯 TUJUAN KESEHATAN")
    tujuan = st.selectbox(
        "Pilih Tujuan Utama:",
        ["Menurunkan Berat Badan", "Menjaga Berat Badan", "Meningkatkan Massa Otot", "Meningkatkan Energi"]
    )
    
    aktivitas = st.selectbox(
        "Tingkat Aktivitas:",
        ["Sedentary (Minim)", "Ringan", "Sedang", "Aktif", "Sangat Aktif"]
    )
    
    st.markdown("### 🔧 PENGATURAN KLUSTER")
    n_clusters = st.slider("Jumlah Klaster", min_value=2, max_value=6, value=4)
    
    if st.button("🔄 PROSES ULANG", type="primary"):
        st.rerun()

# ====================== FUNGSI BANTU ======================
def hitung_skor_kesehatan(calories, fat, protein, sugar, sodium):
    """Hitung skor kesehatan dari 0-100"""
    skor = 100
    
    # Pengurangan untuk faktor tidak sehat
    if calories > 800: skor -= 25
    elif calories > 500: skor -= 15
    elif calories > 300: skor -= 5
    
    if fat > 30: skor -= 20
    elif fat > 20: skor -= 10
    
    if sugar > 30: skor -= 15
    elif sugar > 20: skor -= 8
    
    if sodium > 1500: skor -= 15
    elif sodium > 1000: skor -= 8
    
    # Penambahan untuk faktor sehat
    if protein > 25: skor += 15
    elif protein > 15: skor += 10
    
    return max(0, min(100, skor))

def hitung_kebutuhan_kalori(usia, jenis_kelamin, berat, tinggi, aktivitas, tujuan):
    """Hitung kebutuhan kalori harian"""
    # BMR (Basal Metabolic Rate)
    if jenis_kelamin == "Laki-laki":
        bmr = 88.362 + (13.397 * berat) + (4.799 * tinggi) - (5.677 * usia)
    else:
        bmr = 447.593 + (9.247 * berat) + (3.098 * tinggi) - (4.330 * usia)
    
    # Faktor aktivitas
    faktor = {
        "Sedentary (Minim)": 1.2,
        "Ringan": 1.375,
        "Sedang": 1.55,
        "Aktif": 1.725,
        "Sangat Aktif": 1.9
    }
    
    tdee = bmr * faktor[aktivitas]
    
    # Adjust berdasarkan tujuan
    if tujuan == "Menurunkan Berat Badan":
        target = tdee * 0.85
    elif tujuan == "Meningkatkan Massa Otot":
        target = tdee * 1.15
    else:
        target = tdee
    
    return {
        'BMR': round(bmr),
        'TDEE': round(tdee),
        'Target': round(target),
        'Protein': round((target * 0.3) / 4),
        'Lemak': round((target * 0.25) / 9),
        'Karbohidrat': round((target * 0.45) / 4)
    }

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    """Load dan proses data"""
    try:
        df = pd.read_csv('dataset/fastfood.csv')
        
        # Hitung skor kesehatan untuk setiap menu
        df['health_score'] = df.apply(
            lambda row: hitung_skor_kesehatan(
                row['calories'], 
                row['total_fat'], 
                row['protein'], 
                row['sugar'], 
                row['sodium']
            ), axis=1
        )
        
        # Kategorikan skor kesehatan
        def kategori_sehat(score):
            if score >= 80: return "Sangat Sehat"
            elif score >= 60: return "Sehat"
            elif score >= 40: return "Cukup Sehat"
            elif score >= 20: return "Kurang Sehat"
            else: return "Tidak Sehat"
        
        df['health_category'] = df['health_score'].apply(kategori_sehat)
        
        return df
    except Exception as e:
        st.error(f"❌ ERROR: {str(e)}")
        st.info("⚠️ Pastikan file 'fastfood.csv' ada di folder 'dataset/'")
        return pd.DataFrame()

# Load data
with st.spinner("📥 Memuat dataset..."):
    df = load_data()

if df.empty:
    st.stop()

# ====================== BAGIAN 1: OVERVIEW DATA ======================
st.markdown('<h2 class="section-header">📊 OVERVIEW DATASET</h2>', unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Menu", len(df), "item")
with col2:
    st.metric("Jumlah Restoran", df['restaurant'].nunique())
with col3:
    avg_cal = df['calories'].mean()
    st.metric("Rata Kalori", f"{avg_cal:.0f}", "kcal")
with col4:
    avg_health = df['health_score'].mean()
    st.metric("Skor Kesehatan", f"{avg_health:.1f}", "/100")

# Tampilkan data
with st.expander("🔍 LIHAT DATA (10 baris pertama)", expanded=False):
    st.dataframe(df[['restaurant', 'item', 'calories', 'protein', 'total_fat', 'health_score']].head(10), 
                 use_container_width=True)

# ====================== BAGIAN 2: ANALISIS EKSPLORATIF ======================
st.markdown('<h2 class="section-header">📈 ANALISIS DATA</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Distribusi", "🏪 Per Restoran", "🏆 Menu Terbaik"])

with tab1:
    # Histogram distribusi
    fig = px.histogram(df, x='calories', nbins=30, 
                       title='Distribusi Kalori Menu Fast Food',
                       labels={'calories': 'Kalori', 'count': 'Jumlah Menu'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Kalori per restoran
    avg_by_resto = df.groupby('restaurant').agg({
        'calories': 'mean',
        'health_score': 'mean',
        'item': 'count'
    }).round(1).sort_values('calories', ascending=False)
    
    fig = px.bar(avg_by_resto, x=avg_by_resto.index, y='calories',
                 title='Rata-rata Kalori per Restoran',
                 labels={'x': 'Restoran', 'calories': 'Kalori'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # 10 menu paling sehat
    top_healthy = df.nlargest(10, 'health_score')[['item', 'restaurant', 'calories', 'protein', 'health_score']]
    
    fig = px.bar(top_healthy, x='health_score', y='item',
                 orientation='h',
                 title='Top 10 Menu Paling Sehat',
                 labels={'health_score': 'Skor Kesehatan', 'item': 'Menu'},
                 color='health_score',
                 color_continuous_scale='Greens')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ====================== BAGIAN 3: CLUSTERING K-MEANS ======================
st.markdown('<h2 class="section-header">🔬 KLUSTERISASI K-MEANS</h2>', unsafe_allow_html=True)

# Pilih fitur
feature_options = ['calories', 'total_fat', 'sat_fat', 'cholesterol', 'sodium', 
                   'total_carb', 'sugar', 'protein', 'health_score']
selected_features = st.multiselect(
    "Pilih fitur untuk clustering:",
    feature_options,
    default=['calories', 'total_fat', 'protein', 'health_score']
)

if len(selected_features) < 2:
    st.warning("⚠️ Pilih minimal 2 fitur untuk clustering")
else:
    # Persiapan data
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    with st.spinner("🚀 Menjalankan K-Means Clustering..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Hitung silhouette score
        silhouette = silhouette_score(X_scaled, df['cluster'])
    
    # Tampilkan hasil
    col_score1, col_score2 = st.columns(2)
    with col_score1:
        st.metric("Silhouette Score", f"{silhouette:.3f}")
    with col_score2:
        st.metric("Inertia", f"{kmeans.inertia_:.0f}")
    
    # Visualisasi dengan PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    viz_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': df['cluster'],
        'Menu': df['item'],
        'Restoran': df['restaurant'],
        'Kalori': df['calories'],
        'Skor_Kesehatan': df['health_score']
    })
    
    fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster',
                     hover_data=['Menu', 'Restoran', 'Kalori', 'Skor_Kesehatan'],
                     title='Visualisasi Klaster (PCA)',
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistik per klaster
    st.markdown("#### 📊 STATISTIK PER KLASTER")
    cluster_stats = df.groupby('cluster').agg({
        'calories': ['mean', 'min', 'max'],
        'protein': 'mean',
        'total_fat': 'mean',
        'health_score': 'mean',
        'item': 'count'
    }).round(1)
    
    cluster_stats.columns = ['Kalori_Rata', 'Kalori_Min', 'Kalori_Max', 
                            'Protein_Rata', 'Lemak_Rata', 'SkorSehat_Rata', 'Jumlah_Menu']
    
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Detail per klaster
    st.markdown("#### 🎯 DETAIL PER KLASTER")
    selected_cluster = st.selectbox("Pilih klaster:", sorted(df['cluster'].unique()))
    
    cluster_data = df[df['cluster'] == selected_cluster]
    
    col_cl1, col_cl2, col_cl3 = st.columns(3)
    with col_cl1:
        st.metric("Jumlah Menu", len(cluster_data))
    with col_cl2:
        st.metric("Rata Kalori", f"{cluster_data['calories'].mean():.0f} kcal")
    with col_cl3:
        st.metric("Rata Skor", f"{cluster_data['health_score'].mean():.1f}/100")
    
    # Tampilkan menu dalam klaster
    st.dataframe(
        cluster_data[['item', 'restaurant', 'calories', 'protein', 'total_fat', 'health_score', 'health_category']]
        .sort_values('health_score', ascending=False)
        .head(15),
        use_container_width=True,
        height=300
    )

# ====================== BAGIAN 4: SISTEM REKOMENDASI ======================
st.markdown('<h2 class="section-header">💡 SISTEM REKOMENDASI</h2>', unsafe_allow_html=True)

# Hitung kebutuhan kalori
kebutuhan = hitung_kebutuhan_kalori(usia, jenis_kelamin, berat_badan, tinggi_badan, aktivitas, tujuan)

# Tampilkan kebutuhan
st.markdown("#### 📋 KEBUTUHAN NUTRISI HARIAN")
col_need1, col_need2, col_need3, col_need4 = st.columns(4)
with col_need1:
    st.metric("TDEE", f"{kebutuhan['TDEE']} kcal")
with col_need2:
    st.metric("Target Harian", f"{kebutuhan['Target']} kcal")
with col_need3:
    st.metric("Protein", f"{kebutuhan['Protein']} g")
with col_need4:
    st.metric("Lemak", f"{kebutuhan['Lemak']} g")

# Tombol rekomendasi
if st.button("🎯 DAPATKAN REKOMENDASI MENU", type="primary", use_container_width=True):
    with st.spinner("🔍 Mencari menu terbaik..."):
        if 'cluster' in df.columns:
            # Cari klaster terbaik berdasarkan tujuan
            if tujuan == "Menurunkan Berat Badan":
                # Klaster dengan kalori rendah, skor tinggi
                cluster_scores = df.groupby('cluster').apply(
                    lambda x: (1000 - x['calories'].mean()) * 0.7 + x['health_score'].mean() * 0.3
                )
                best_cluster = cluster_scores.idxmax()
                
            elif tujuan == "Meningkatkan Massa Otot":
                # Klaster dengan protein tinggi
                cluster_scores = df.groupby('cluster')['protein'].mean()
                best_cluster = cluster_scores.idxmax()
                
            elif tujuan == "Meningkatkan Energi":
                # Klaster dengan karbohidrat moderat
                cluster_scores = df.groupby('cluster')['total_carb'].mean()
                # Cari yang paling mendekati 60g (moderat)
                best_cluster = (cluster_scores - 60).abs().idxmin()
                
            else:  # Menjaga Berat Badan
                # Klaster dengan skor kesehatan tertinggi
                cluster_scores = df.groupby('cluster')['health_score'].mean()
                best_cluster = cluster_scores.idxmax()
            
            # Ambil 6 menu terbaik dari klaster tersebut
            recommendations = df[df['cluster'] == best_cluster]
            recommendations = recommendations.nlargest(6, 'health_score')
            
            st.success(f"✅ Ditemukan {len(recommendations)} menu dari Klaster {best_cluster}")
            
            # Tampilkan rekomendasi
            st.markdown("#### 🍽️ MENU YANG DIREKOMENDASIKAN")
            
            cols = st.columns(3)
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx % 3]:
                    # Tentukan class kesehatan
                    if row['health_score'] >= 70:
                        health_class = "health-good"
                    elif row['health_score'] >= 50:
                        health_class = "health-moderate"
                    else:
                        health_class = "health-poor"
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{row['item']}</h4>
                        <p><strong>🏪 Restoran:</strong> {row['restaurant']}</p>
                        <p><strong>🔥 Kalori:</strong> {row['calories']} kcal</p>
                        <p><strong>💪 Protein:</strong> {row['protein']}g</p>
                        <p><strong>🏷️ Kategori:</strong> {row['health_category']}</p>
                        <p><strong>⭐ Skor:</strong> <span class="{health_class}">{row['health_score']}/100</span></p>
                        <p><strong>🔢 Klaster:</strong> {row['cluster']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Rencana makan harian
            st.markdown("#### 📅 CONTOH POLA MAKAN HARIAN")
            
            meals = [
                ("Sarapan", "07:00", "Menu tinggi protein", kebutuhan['Target'] * 0.25),
                ("Snack Pagi", "10:00", "Buah atau yogurt", kebutuhan['Target'] * 0.1),
                ("Makan Siang", "13:00", "Menu seimbang", kebutuhan['Target'] * 0.35),
                ("Snack Sore", "16:00", "Kacang atau smoothie", kebutuhan['Target'] * 0.1),
                ("Makan Malam", "19:00", "Menu ringan", kebutuhan['Target'] * 0.2),
            ]
            
            for meal, time, desc, calories in meals:
                col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1, 2, 1])
                with col_m1:
                    st.write(f"**{meal}**")
                with col_m2:
                    st.write(time)
                with col_m3:
                    st.write(desc)
                with col_m4:
                    st.write(f"**{calories:.0f} kcal**")
        else:
            st.warning("⚠️ Lakukan clustering terlebih dahulu!")

# ====================== BAGIAN 5: EKSPOR HASIL ======================
st.markdown('<h2 class="section-header">💾 EKSPOR HASIL</h2>', unsafe_allow_html=True)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if st.button("📥 Download Data dengan Klaster"):
        if 'cluster' in df.columns:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Klik untuk download CSV",
                data=csv,
                file_name="fastfood_with_clusters.csv",
                mime="text/csv"
            )
        else:
            st.warning("Lakukan clustering terlebih dahulu")

with col_exp2:
    if st.button("📊 Download Statistik"):
        if 'cluster' in df.columns:
            stats = df.groupby('cluster').agg({
                'calories': ['mean', 'min', 'max'],
                'protein': 'mean',
                'health_score': 'mean',
                'item': 'count'
            }).round(2)
            
            csv = stats.to_csv().encode('utf-8')
            st.download_button(
                label="Klik untuk download",
                data=csv,
                file_name="cluster_statistics.csv",
                mime="text/csv"
            )
        else:
            st.warning("Lakukan clustering terlebih dahulu")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <h3 style="color: #2E8B57;">🍎 SISTEM REKOMENDASI POLA MAKAN SEHAT</h3>
    <p><strong>Teknologi:</strong> Streamlit • Scikit-learn • Pandas • Plotly</p>
    <p><strong>Metode:</strong> K-Means Clustering • PCA • Analisis Nutrisi</p>
    <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">
        ⚠️ <strong>Disclaimer:</strong> Rekomendasi ini bersifat informatif dan edukatif. 
        Konsultasikan dengan ahli gizi untuk kebutuhan kesehatan spesifik.
    </p>
</div>
""", unsafe_allow_html=True)