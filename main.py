import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter
from wordcloud import WordCloud

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title = "Breakup Emotion Dashboard",
    page_icon  = "💔",
    layout     = "wide"
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_excel("Emotion_Dataset_Merged.xlsx")
    return df

df = load_data()

# ==================================================
# KONSTANTA
# ==================================================
EMOTION_COLORS = {
    "anger"      : "#E05C5C",
    "anxiety"    : "#F0A500",
    "acceptance" : "#4CAF82",
}

STOPWORDS = {
    "i", "me", "my", "he", "she", "we", "they", "you",
    "it", "is", "am", "are", "was", "were", "be", "been",
    "the", "a", "an", "and", "or", "but", "so", "if",
    "to", "of", "in", "on", "at", "for", "with", "about",
    "that", "this", "not", "no", "do", "did", "have",
    "has", "had", "will", "would", "can", "could", "just",
    "get", "got", "go", "going", "feel", "feeling", "felt",
    "know", "want", "need", "like", "make", "think",
    "really", "very", "so", "much", "more", "still",
    "him", "her", "his", "its", "our", "their", "us",
    "from", "when", "how", "what", "why", "who", "which",
    "there", "then", "than", "now", "back", "one", "all",
    "been", "being", "some", "even", "ever", "never",
    "time", "way", "out", "up", "down", "over", "after",
    "also", "well", "good", "bad", "too", "again", "away",
    "said", "say", "see", "look", "come", "came",
}

WC_COLORMAPS = {
    "anger"      : "Reds",
    "anxiety"    : "YlOrBr",
    "acceptance" : "Greens",
}

# ==================================================
# HEADER
# ==================================================
st.title("💔 Breakup Emotion Dashboard")
st.markdown(
    "Dashboard analisis emosi pada teks putus cinta "
    "berdasarkan tiga kelas: **Anger**, **Anxiety**, dan **Acceptance**."
)
st.divider()

# ==================================================
# SIDEBAR — FILTER
# ==================================================
st.sidebar.header("⚙️ Filter Data")

# filter confidence
min_conf      = float(df["emotion_confidence"].min())
max_conf      = float(df["emotion_confidence"].max())
conf_threshold = st.sidebar.slider(
    "Minimum Confidence",
    min_value = round(min_conf, 2),
    max_value = round(max_conf, 2),
    value     = round(min_conf, 2),
    step      = 0.01
)

# filter emosi
emotion_options  = ["Semua"] + sorted(df["predicted_emotion"].unique().tolist())
selected_emotion_filter = st.sidebar.selectbox(
    "Filter Emosi",
    options = emotion_options,
    index   = 0
)

# terapkan filter
df_filtered = df[df["emotion_confidence"] >= conf_threshold].copy()

if selected_emotion_filter != "Semua":
    df_filtered = df_filtered[
        df_filtered["predicted_emotion"] == selected_emotion_filter
    ]

st.sidebar.markdown(f"**Total data:** `{len(df_filtered):,}` baris")

# ==================================================
# SECTION 1 — METRIK RINGKASAN
# ==================================================
st.subheader("📊 Ringkasan Data")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Data", f"{len(df_filtered):,}")
with col2:
    st.metric(
        "Rata-rata Confidence",
        f"{df_filtered['emotion_confidence'].mean():.2%}"
    )
with col3:
    st.metric(
        "Rata-rata Anger Prob",
        f"{df_filtered['anger_prob'].mean():.2%}"
    )
with col4:
    dominant = (
        df_filtered["predicted_emotion"].mode()[0]
        if len(df_filtered) > 0 else "-"
    )
    st.metric("Emosi Dominan", dominant.capitalize())

st.divider()

# ==================================================
# SECTION 2 — DISTRIBUSI EMOSI
# ==================================================
st.subheader("📈 Distribusi Label Emosi")

emotion_counts = df_filtered["predicted_emotion"].value_counts()
emotion_labels = emotion_counts.index.tolist()
emotion_values = emotion_counts.values.tolist()
bar_colors     = [EMOTION_COLORS.get(e, "#888888") for e in emotion_labels]

col_left, col_right = st.columns(2)

# bar chart jumlah
with col_left:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(
        emotion_labels, emotion_values,
        color=bar_colors, edgecolor="white", width=0.5
    )
    ax1.set_title("Jumlah Data per Emosi", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Emosi")
    ax1.set_ylabel("Jumlah")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    for bar, val in zip(bars, emotion_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(emotion_values) * 0.01,
            f"{val:,}", ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )
    ax1.set_ylim(0, max(emotion_values) * 1.15)
    ax1.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig1)

# bar chart persentase
with col_right:
    total = sum(emotion_values)
    pcts  = [v / total * 100 for v in emotion_values]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars2 = ax2.bar(
        emotion_labels, pcts,
        color=bar_colors, edgecolor="white", width=0.5
    )
    ax2.set_title("Persentase Data per Emosi", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Emosi")
    ax2.set_ylabel("Persentase (%)")
    for bar, pct in zip(bars2, pcts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(pcts) * 0.01,
            f"{pct:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )
    ax2.set_ylim(0, max(pcts) * 1.15)
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)

st.divider()

# ==================================================
# SECTION 3 — PROBABILITAS RATA-RATA
# ==================================================
st.subheader("🎯 Rata-rata Probabilitas per Emosi")

col_a, col_b, col_c = st.columns(3)

avg_anger      = df_filtered["anger_prob"].mean()
avg_anxiety    = df_filtered["anxiety_prob"].mean()
avg_acceptance = df_filtered["acceptance_prob"].mean()

with col_a:
    st.metric("😡 Anger",      f"{avg_anger:.2%}")
    st.progress(float(avg_anger))
with col_b:
    st.metric("😰 Anxiety",    f"{avg_anxiety:.2%}")
    st.progress(float(avg_anxiety))
with col_c:
    st.metric("🌿 Acceptance", f"{avg_acceptance:.2%}")
    st.progress(float(avg_acceptance))

st.divider()

# ==================================================
# SECTION 4 — WORDCLOUD & TOP WORDS
# ==================================================
st.subheader("☁️ Word Cloud & Kata Terbanyak per Emosi")

selected_emotion = st.selectbox(
    "Pilih Emosi",
    options     = ["anger", "anxiety", "acceptance"],
    format_func = lambda x: x.capitalize()
)

df_emotion_sel = df_filtered[
    df_filtered["predicted_emotion"] == selected_emotion
]

if len(df_emotion_sel) == 0:
    st.warning("Tidak ada data untuk emosi ini dengan filter yang dipilih.")
else:
    corpus     = " ".join(df_emotion_sel["input_no_punct"].dropna().tolist())
    tokens     = [t for t in corpus.split() if t not in STOPWORDS and len(t) > 2]
    token_freq = Counter(tokens)

    TOP_N = st.slider("Jumlah Top Kata", min_value=5, max_value=30, value=15)

    col_wc, col_bar = st.columns(2)

    # wordcloud
    with col_wc:
        st.markdown(f"**Word Cloud — {selected_emotion.capitalize()}**")
        wc = WordCloud(
            width            = 800,
            height           = 500,
            background_color = "white",
            colormap         = WC_COLORMAPS[selected_emotion],
            max_words        = 100,
            stopwords        = STOPWORDS,
            collocations     = False,
        ).generate_from_frequencies(token_freq)

        fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # top words bar chart
    with col_bar:
        st.markdown(f"**Top {TOP_N} Kata — {selected_emotion.capitalize()}**")
        top_words = sorted(
            token_freq.items(), key=lambda x: x[1], reverse=True
        )[:TOP_N]
        words = [w for w, _ in top_words][::-1]
        freqs = [f for _, f in top_words][::-1]

        fig_bar, ax_bar = plt.subplots(figsize=(6, TOP_N * 0.4 + 1))
        bars_bar = ax_bar.barh(
            words, freqs,
            color     = EMOTION_COLORS[selected_emotion],
            edgecolor = "white"
        )
        ax_bar.set_xlabel("Frekuensi")
        ax_bar.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        for bar, freq in zip(bars_bar, freqs):
            ax_bar.text(
                bar.get_width() + max(freqs) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{freq:,}", va="center", fontsize=9
            )
        ax_bar.set_xlim(0, max(freqs) * 1.15)
        ax_bar.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig_bar)

st.divider()

# ==================================================
# SECTION 5 — DISTRIBUSI CONFIDENCE
# ==================================================
st.subheader("📉 Distribusi Confidence Score per Emosi")

fig_conf, ax_conf = plt.subplots(figsize=(10, 4))
for emotion in ["anger", "anxiety", "acceptance"]:
    data = df_filtered[
        df_filtered["predicted_emotion"] == emotion
    ]["emotion_confidence"]
    ax_conf.hist(
        data, bins=30, alpha=0.6,
        label     = emotion.capitalize(),
        color     = EMOTION_COLORS[emotion],
        edgecolor = "white"
    )
ax_conf.set_xlabel("Confidence Score")
ax_conf.set_ylabel("Jumlah")
ax_conf.legend()
ax_conf.spines[["top", "right"]].set_visible(False)
st.pyplot(fig_conf)

st.divider()

# ==================================================
# SECTION 6 — EKSPLORASI DATA
# ==================================================
st.subheader("🔍 Eksplorasi Data")

search_keyword = st.text_input(
    "Cari Kata dalam Teks",
    placeholder = "contoh: heartbroken"
)

df_explore = df_filtered.copy()

if search_keyword:
    df_explore = df_explore[
        df_explore["input_clean"].str.contains(
            search_keyword, case=False, na=False
        )
    ]

st.markdown(f"Menampilkan **{len(df_explore):,}** baris")

st.dataframe(
    df_explore[[
        "input_clean", "input_no_punct",
        "anger_prob", "anxiety_prob", "acceptance_prob",
        "predicted_emotion", "emotion_confidence"
    ]].reset_index(drop=True),
    use_container_width = True,
    height              = 400
)

# tombol download
csv = df_explore.to_csv(index=False).encode("utf-8")
st.download_button(
    label     = "⬇️ Download Data Terfilter (CSV)",
    data      = csv,
    file_name = "data_filtered.csv",
    mime      = "text/csv"
)

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("💔 Breakup Emotion Dashboard — dibuat dengan Streamlit")
