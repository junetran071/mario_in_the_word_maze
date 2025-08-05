import streamlit as st
import pandas as pd
import io

# --- Mario Theme Styling ---
mario_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

html, body {
    background-image: url('https://www.transparenttextures.com/patterns/stardust.png'), 
                      url('https://wallpapercave.com/wp/wp2854319.png');
    background-size: cover;
    font-family: 'Press Start 2P', cursive;
    color: #fff;
}
h1, h2, h3 {
    color: #ffd60a;
    text-shadow: 2px 2px #000;
}
.stButton>button {
    background-color: #ff0000;
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
    border: 3px solid yellow;
    box-shadow: 2px 2px #000;
    font-family: 'Press Start 2P', cursive;
}
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    border: 2px solid #ffd60a !important;
    background-color: #fff3cd !important;
    color: #000 !important;
    font-weight: bold !important;
    font-family: 'Press Start 2P', cursive;
}
.stFileUploader>div>div {
    background-color: #90e0ef;
    border: 2px dashed #03045e;
    padding: 10px;
}
</style>
"""

coin_sound = """
<audio id="coinSound" src="https://www.myinstants.com/media/sounds/mario-coin.mp3"></audio>
<script>
function playCoinSound() {
    document.getElementById('coinSound').play();
}
</script>
"""

st.set_page_config(page_title="Super Mario Classifier Metrics", layout="centered")
st.markdown(mario_css, unsafe_allow_html=True)
st.markdown(coin_sound, unsafe_allow_html=True)

# 🎯 Predefined Classifiers
classifiers = [
    "urgency_marketing", "exclusive_marketing", "personal_milestone",
    "gratitude_reflection", "local_business", "social_proof", "discount_pricing"
]

# Title
st.title("🍄 Super Mario Classifier Metrics")
st.write("🏰 Help Mario analyze your Instagram posts using these fixed classifier categories!")

# Step 1: Upload CSV
st.header("🧱 1. Upload Your Data")
uploaded_file = st.file_uploader("📁 Upload your Instagram CSV data", type=["csv"])

if uploaded_file:
    try:
        decoded = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded))
    except UnicodeDecodeError:
        decoded = uploaded_file.getvalue().decode('latin1')
        df = pd.read_csv(io.StringIO(decoded))

    # Step 2: Select the text column
    st.header("🍄 2. Choose Text Column")
    text_column = st.selectbox("📝 Select the column that contains the post text", df.columns.tolist())

    # Step 3: Process
    st.header("🚀 3. Generate Metrics")

    if st.button("🔊 Generate Statement-Level Metrics"):
        st.markdown("<script>playCoinSound()</script>", unsafe_allow_html=True)

        metrics_df = df.copy()
        found_classifiers = []
        missing_classifiers = []

        for classifier in classifiers:
            if classifier in df.columns:
                found_classifiers.append(classifier)
                metrics_df[f"{classifier}_word_count"] = df.apply(
                    lambda row: len(str(row[text_column]).split()) if row[classifier] == 1 else 0,
                    axis=1
                )
                metrics_df[f"{classifier}_word_pct"] = df.apply(
                    lambda row: (len(str(row[text_column]).split()) if row[classifier] == 1 else 0) / max(len(str(row[text_column]).split()), 1),
                    axis=1
                )
            else:
                missing_classifiers.append(classifier)

        if missing_classifiers:
            st.warning(f"🚫 These classifier columns were not found: {', '.join(missing_classifiers)}")

        if found_classifiers:
            st.subheader("📊 Final Score: Statement Metrics")
            st.dataframe(metrics_df)
        else:
            st.error("No valid classifier columns found in your dataset.")

    if st.button("👑 Boss Level: Aggregate by ID"):
        st.markdown("<script>playCoinSound()</script>", unsafe_allow_html=True)

        if "ID" not in df.columns:
            st.warning("⚠️ No 'ID' column found for aggregation.")
        else:
            valid_classifiers = [c for c in classifiers if c in df.columns]
            if valid_classifiers:
                agg_df = df.groupby("ID")[valid_classifiers].sum().reset_index()
                st.subheader("🏁 Aggregated Metrics by ID")
                st.dataframe(agg_df)
            else:
                st.warning("⚠️ None of the predefined classifier columns exist in your data.")
else:
    st.info("🧭 Upload your CSV to begin the adventure!")
