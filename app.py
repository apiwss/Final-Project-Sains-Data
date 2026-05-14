import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load model & stats ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("car_price_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_stats():
    with open("model_stats.json") as f:
        return json.load(f)

model = load_model()
stats = load_stats()

FEATURES = ['Engine_size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
            'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency', 'Power_perf_factor']

FEATURE_LABELS = {
    'Engine_size':       'Engine Size (Liter)',
    'Horsepower':        'Horsepower (HP)',
    'Wheelbase':         'Wheelbase (inch)',
    'Width':             'Width (inch)',
    'Length':            'Length (inch)',
    'Curb_weight':       'Curb Weight (ton)',
    'Fuel_capacity':     'Fuel Capacity (gallon)',
    'Fuel_efficiency':   'Fuel Efficiency (MPG)',
    'Power_perf_factor': 'Power Perf. Factor',
}

FEATURE_ICONS = {
    'Engine_size': '🔧', 'Horsepower': '⚡', 'Wheelbase': '📏',
    'Width': '↔️', 'Length': '↕️', 'Curb_weight': '⚖️',
    'Fuel_capacity': '🛢️', 'Fuel_efficiency': '⛽', 'Power_perf_factor': '🏎️',
}

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 50%, #0288d1 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 20px rgba(21,101,192,0.35);
  }
  .header-banner h1 { margin: 0; font-size: 2rem; font-weight: 800; }
  .header-banner p  { margin: 6px 0 0; font-size: 1rem; opacity: 0.88; }

  /* Result card */
  .result-card {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border: 2px solid #43a047;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 2px 12px rgba(67,160,71,0.2);
  }
  .result-card .label { font-size: 0.95rem; color: #558b2f; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
  .result-card .price { font-size: 3rem; font-weight: 900; color: #2e7d32; line-height: 1.1; margin: 8px 0; }
  .result-card .sub   { font-size: 1rem; color: #689f38; }

  /* Metric cards */
  .metric-row { display: flex; gap: 12px; margin: 12px 0; }
  .metric-card {
    flex: 1;
    background: white;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .metric-card .m-label { font-size: 0.78rem; color: #757575; font-weight: 600; text-transform: uppercase; }
  .metric-card .m-value { font-size: 1.5rem; font-weight: 800; color: #1565c0; margin-top: 4px; }

  /* Section title */
  .section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a237e;
    border-left: 4px solid #1565c0;
    padding-left: 10px;
    margin: 20px 0 12px;
  }

  /* Spec row */
  .spec-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  .spec-table td { padding: 6px 8px; border-bottom: 1px solid #f0f0f0; }
  .spec-table td:first-child { color: #616161; }
  .spec-table td:last-child  { font-weight: 700; color: #212121; text-align: right; }

  /* Footer */
  .footer {
    background: #1a237e;
    color: white;
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    margin-top: 20px;
    font-size: 0.88rem;
    opacity: 0.9;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #f5f7ff; }
  .sidebar-card {
    background: white;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border: 1px solid #e8eaf6;
    font-size: 0.85rem;
  }
  .sidebar-card b { color: #1a237e; }

  /* Tabs */
  div[data-testid="stTab"] button { font-weight: 600; }

  /* Button */
  div.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0288d1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    font-size: 1.05rem;
    font-weight: 700;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
    box-shadow: 0 3px 10px rgba(21,101,192,0.4);
  }
  div.stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div style="font-size:3.5rem">🚗</div>
  <div>
    <h1>Sistem Prediksi Harga Mobil</h1>
    <p>Masukkan spesifikasi teknis kendaraan untuk mendapatkan estimasi harga pasar secara otomatis menggunakan Machine Learning</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar – model info ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Informasi Model")

    r2_pct = stats['r2'] * 100
    rmse_usd = stats['rmse'] * 1000

    st.markdown(f"""
    <div class="sidebar-card">
      <b>Algoritma</b><br>Linear Regression
    </div>
    <div class="sidebar-card">
      <b>R² Score</b><br>
      <span style="font-size:1.4rem;color:#1565c0;font-weight:900">{r2_pct:.2f}%</span><br>
      <span style="font-size:0.78rem;color:#888">Akurasi model menjelaskan variasi harga</span>
    </div>
    <div class="sidebar-card">
      <b>RMSE</b><br>
      <span style="font-size:1.4rem;color:#e65100;font-weight:900">${rmse_usd:,.0f}</span><br>
      <span style="font-size:0.78rem;color:#888">Rata-rata error prediksi</span>
    </div>
    <div class="sidebar-card">
      <b>Jumlah Fitur</b><br>9 variabel teknis
    </div>
    <div class="sidebar-card">
      <b>Split Data</b><br>80% Training / 20% Testing
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Petunjuk Penggunaan")
    st.info(
        "1. Isi spesifikasi mobil di form\n"
        "2. Klik **Hitung Harga Mobil**\n"
        "3. Lihat hasil prediksi di panel kanan\n"
        "4. Eksplorasi tab **Analisis Pasar** untuk insight data"
    )

# ─── Main tabs ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Prediksi Harga", "📈 Analisis Pasar"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    # ── LEFT: Input form ──────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="section-title">📋 Input Spesifikasi Mobil</div>', unsafe_allow_html=True)

        mins  = stats['feature_mins']
        maxs  = stats['feature_maxs']
        means = stats['feature_means']

        # Preset buttons
        st.markdown("**🎯 Pilih Preset Spesifikasi:**")
        p1, p2, p3, p4 = st.columns(4)
        preset = None
        if p1.button("🏃 Ekonomis"):   preset = "ekonomis"
        if p2.button("🚙 Menengah"):   preset = "menengah"
        if p3.button("🏆 Premium"):    preset = "premium"
        if p4.button("🔄 Reset"):      preset = "reset"

        PRESETS = {
            "ekonomis": dict(Engine_size=1.8, Horsepower=120, Wheelbase=100.0,
                             Width=68.0, Length=175.0, Curb_weight=2.8,
                             Fuel_capacity=13.0, Fuel_efficiency=30.0, Power_perf_factor=55.0),
            "menengah": dict(Engine_size=2.3, Horsepower=150, Wheelbase=105.0,
                             Width=70.0, Length=185.0, Curb_weight=3.2,
                             Fuel_capacity=15.0, Fuel_efficiency=26.0, Power_perf_factor=70.0),
            "premium":  dict(Engine_size=3.5, Horsepower=250, Wheelbase=112.0,
                             Width=74.0, Length=195.0, Curb_weight=4.0,
                             Fuel_capacity=20.0, Fuel_efficiency=20.0, Power_perf_factor=110.0),
            "reset":    {f: means[f] for f in FEATURES},
        }

        def get_val(key):
            if preset and preset in PRESETS:
                return PRESETS[preset][key]
            return st.session_state.get(f"inp_{key}", means[key])

        st.markdown("---")

        # Form inputs
        c1, c2 = st.columns(2)
        with c1:
            engine   = st.number_input("🔧 Engine Size (L)",    min_value=float(mins['Engine_size']),    max_value=float(maxs['Engine_size']),    value=float(get_val('Engine_size')),    step=0.1,  key="inp_Engine_size",    format="%.1f")
            hp       = st.number_input("⚡ Horsepower (HP)",     min_value=float(mins['Horsepower']),     max_value=float(maxs['Horsepower']),     value=float(get_val('Horsepower')),     step=5.0,  key="inp_Horsepower",     format="%.0f")
            wheel    = st.number_input("📏 Wheelbase (inch)",    min_value=float(mins['Wheelbase']),      max_value=float(maxs['Wheelbase']),      value=float(get_val('Wheelbase')),      step=0.5,  key="inp_Wheelbase",      format="%.1f")
            width    = st.number_input("↔️ Width (inch)",         min_value=float(mins['Width']),          max_value=float(maxs['Width']),          value=float(get_val('Width')),          step=0.5,  key="inp_Width",          format="%.1f")
            length   = st.number_input("↕️ Length (inch)",        min_value=float(mins['Length']),         max_value=float(maxs['Length']),         value=float(get_val('Length')),         step=0.5,  key="inp_Length",         format="%.1f")
        with c2:
            cweight  = st.number_input("⚖️ Curb Weight (ton)",   min_value=float(mins['Curb_weight']),    max_value=float(maxs['Curb_weight']),    value=float(get_val('Curb_weight')),    step=0.1,  key="inp_Curb_weight",    format="%.1f")
            fcap     = st.number_input("🛢️ Fuel Capacity (gal)", min_value=float(mins['Fuel_capacity']),  max_value=float(maxs['Fuel_capacity']),  value=float(get_val('Fuel_capacity')),  step=0.5,  key="inp_Fuel_capacity",  format="%.1f")
            feff     = st.number_input("⛽ Fuel Efficiency (MPG)",min_value=float(mins['Fuel_efficiency']),max_value=float(maxs['Fuel_efficiency']),value=float(get_val('Fuel_efficiency')),step=1.0,  key="inp_Fuel_efficiency",format="%.0f")
            ppf      = st.number_input("🏎️ Power Perf. Factor",  min_value=float(mins['Power_perf_factor']),max_value=float(maxs['Power_perf_factor']),value=float(get_val('Power_perf_factor')),step=1.0, key="inp_Power_perf_factor", format="%.1f")

        hitung = st.button("🔮 Hitung Harga Mobil", use_container_width=True)

    # ── RIGHT: Result panel ───────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-title">💰 Hasil Prediksi</div>', unsafe_allow_html=True)

        input_vals = [engine, hp, wheel, width, length, cweight, fcap, feff, ppf]
        input_df   = pd.DataFrame([input_vals], columns=FEATURES)
        prediction = model.predict(input_df)[0]
        pred_usd   = prediction * 1000

        # Segment tag
        if prediction < 15:
            segment, seg_color, seg_icon = "Ekonomis", "#43a047", "🟢"
        elif prediction < 25:
            segment, seg_color, seg_icon = "Menengah", "#fb8c00", "🟡"
        elif prediction < 40:
            segment, seg_color, seg_icon = "Premium", "#e53935", "🔴"
        else:
            segment, seg_color, seg_icon = "Mewah / Luxury", "#7b1fa2", "🟣"

        st.markdown(f"""
        <div class="result-card">
          <div class="label">Perkiraan Harga Mobil</div>
          <div class="price">${pred_usd:,.0f}</div>
          <div class="sub">≈ {prediction:.3f}K USD &nbsp;|&nbsp;
            <span style="color:{seg_color};font-weight:700">{seg_icon} Segmen {segment}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        min_p, max_p = 5, 80
        pct = min(max((prediction - min_p) / (max_p - min_p), 0), 1)
        st.markdown(f"""
        <div style="margin: 4px 0 16px;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#888;margin-bottom:4px">
            <span>$5K (Min)</span><span>$80K (Max)</span>
          </div>
          <div style="background:#e0e0e0;border-radius:99px;height:12px;overflow:hidden">
            <div style="width:{pct*100:.1f}%;background:linear-gradient(90deg,#43a047,{seg_color});height:100%;border-radius:99px;transition:width 0.5s"></div>
          </div>
          <div style="text-align:right;font-size:0.78rem;color:{seg_color};margin-top:3px;font-weight:700">
            {pct*100:.0f}% dari rentang pasar
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Spesifikasi ringkasan
        st.markdown('<div class="section-title">📋 Spesifikasi Diinput</div>', unsafe_allow_html=True)
        spec_rows = [
            ("🔧 Engine Size",        f"{engine:.1f} L"),
            ("⚡ Horsepower",          f"{hp:.0f} HP"),
            ("📏 Wheelbase",           f"{wheel:.1f} inch"),
            ("↔️ Width",               f"{width:.1f} inch"),
            ("↕️ Length",              f"{length:.1f} inch"),
            ("⚖️ Curb Weight",         f"{cweight:.1f} ton"),
            ("🛢️ Fuel Capacity",       f"{fcap:.1f} galon"),
            ("⛽ Fuel Efficiency",     f"{feff:.0f} MPG"),
            ("🏎️ Power Perf. Factor",  f"{ppf:.1f}"),
        ]
        rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in spec_rows)
        st.markdown(f'<table class="spec-table">{rows_html}</table>', unsafe_allow_html=True)

        # Radar / konteks perbandingan vs rata-rata
        st.markdown('<div class="section-title">📊 Perbandingan vs Rata-Rata Pasar</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        labels_short = ['Engine', 'HP', 'Whlbase', 'Width', 'Length', 'Weight', 'Fuel Cap', 'Fuel Eff', 'Perf']
        user_vals = [engine, hp, wheel, width, length, cweight, fcap, feff, ppf]
        avg_vals  = [means[f] for f in FEATURES]

        # Normalize to 0-1
        min_v = [mins[f] for f in FEATURES]
        max_v = [maxs[f] for f in FEATURES]
        user_norm = [(v - mn) / (mx - mn) for v, mn, mx in zip(user_vals, min_v, max_v)]
        avg_norm  = [(v - mn) / (mx - mn) for v, mn, mx in zip(avg_vals, min_v, max_v)]

        x = np.arange(len(labels_short))
        w = 0.35
        b1 = ax.bar(x - w/2, user_norm, w, label='Spesifikasi Anda', color='#1565c0', alpha=0.85)
        b2 = ax.bar(x + w/2, avg_norm,  w, label='Rata-rata Pasar',  color='#e0e0e0', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=7, rotation=25, ha='right')
        ax.set_yticks([])
        ax.set_ylabel("Nilai Relatif", fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Footer identity
        st.markdown("""
        <div class="footer">
          <b>SISTEM PREDIKSI HARGA MOBIL</b><br>
          Dibuat menggunakan Machine Learning · Linear Regression<br>
          Dataset: Car Sales · Matakuliah Sains Data
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ANALISIS PASAR
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    top10 = pd.DataFrame(stats['top10'])
    top10['Full_Name'] = top10['Full_Name']

    st.markdown('<div class="section-title">🏆 10 Mobil dengan Penjualan Terbanyak</div>', unsafe_allow_html=True)

    # Chart penjualan
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('none'); ax.set_facecolor('none')
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.85, 10))
    bars = ax.barh(top10['Full_Name'], top10['Sales_in_thousands'], color=colors)
    for bar, val in zip(bars, top10['Sales_in_thousands']):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}K', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Jumlah Penjualan (ribuan unit)')
    ax.set_title('Top 10 Mobil Terlaris', fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # 3 charts berdampingan
    ca, cb, cc = st.columns(3)

    with ca:
        st.markdown('<div class="section-title">💰 Harga (K USD)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor('none'); ax.set_facecolor('none')
        colors2 = plt.cm.Oranges_r(np.linspace(0.3, 0.85, 10))
        bars2 = ax.barh(top10['Full_Name'], top10['Price_in_thousands'], color=colors2)
        for bar, val in zip(bars2, top10['Price_in_thousands']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'${val:.0f}K', va='center', fontsize=8, fontweight='bold')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlabel('Harga (K USD)', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with cb:
        st.markdown('<div class="section-title">⚡ Horsepower (HP)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor('none'); ax.set_facecolor('none')
        colors3 = plt.cm.Reds_r(np.linspace(0.3, 0.85, 10))
        bars3 = ax.barh(top10['Full_Name'], top10['Horsepower'], color=colors3)
        for bar, val in zip(bars3, top10['Horsepower']):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=8, fontweight='bold')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlabel('Horsepower (HP)', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with cc:
        st.markdown('<div class="section-title">⛽ Fuel Efficiency (MPG)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor('none'); ax.set_facecolor('none')
        colors4 = plt.cm.Greens_r(np.linspace(0.3, 0.85, 10))
        bars4 = ax.barh(top10['Full_Name'], top10['Fuel_efficiency'], color=colors4)
        for bar, val in zip(bars4, top10['Fuel_efficiency']):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=8, fontweight='bold')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlabel('Fuel Efficiency (MPG)', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # Tabel data
    st.markdown('<div class="section-title">📋 Tabel Data Top 10 Mobil Terlaris</div>', unsafe_allow_html=True)
    display_df = top10.rename(columns={
        'Full_Name':           'Model Mobil',
        'Sales_in_thousands':  'Penjualan (K)',
        'Price_in_thousands':  'Harga (K USD)',
        'Engine_size':         'Engine (L)',
        'Horsepower':          'HP',
        'Fuel_efficiency':     'MPG',
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Model coefficients insight
    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Pengaruh Setiap Variabel terhadap Harga</div>', unsafe_allow_html=True)
    coef = stats['coef']
    coef_df = pd.DataFrame({
        'Variabel': [FEATURE_LABELS[f] for f in FEATURES],
        'Koefisien': [coef[f] for f in FEATURES],
    }).sort_values('Koefisien', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('none'); ax.set_facecolor('none')
    bar_colors = ['#e53935' if v >= 0 else '#1565c0' for v in coef_df['Koefisien']]
    ax.barh(coef_df['Variabel'], coef_df['Koefisien'], color=bar_colors)
    ax.axvline(0, color='#333', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Nilai Koefisien (dampak per satuan terhadap harga K USD)')
    ax.set_title('Koefisien Regresi Linear — Pengaruh Tiap Fitur terhadap Harga', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    red_patch  = mpatches.Patch(color='#e53935', label='Meningkatkan harga (+)')
    blue_patch = mpatches.Patch(color='#1565c0', label='Menurunkan harga (−)')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption("Koefisien positif (merah) berarti semakin besar nilainya → harga makin tinggi. Koefisien negatif (biru) berarti semakin besar nilainya → harga makin rendah.")
