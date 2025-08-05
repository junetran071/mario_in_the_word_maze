import streamlit as st
import pandas as pd
import io

# 🎨 Super Mario Theme CSS
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
.stTextArea textarea, .stTextInput input {
    border: 2px solid #ffd60a !important;
    background-color: #fff3cd;
    color: #000;
    font-weight: bold;
    font-family: 'Press Start 2P', cursive;
}
.stFileUploader>div>div {
    background-color: #90e0ef;
    border: 2px dashed #03045e;
    padding: 10px;
}
</style>
"""

# 🔊 Coin sound effect
coin_sound = """
<audio id="coinSound" src="https://www.myinstants.com/media/sounds/mario-coin.mp3"></audio>
<script>
function playCoinSound() {
    document.getElementById('coinSound').play();
}
</script>
"""

# ⬅️ Inject styles
st.set_page_config(page_title="Super Mario Classifier Metrics", layout="centered")
st.markdown(mario_css, unsafe_allow_html=True)
st.markdown(coin_sound, unsafe_allow_html=True)

# 🏁 Title
st.title("🍄 Super Mario Classifier Metrics")
st.write("🏰 Help Mario analyze your Instagram posts using classifiers and unlock your data’s hidden power-ups!")

# 🧱 Step 1: Upload CSV
st.header("🧱 1. Upload Your Data")
uploaded_file = st.file_uploader("📁 Upload your Instagram CSV data", type=["csv"])

# 🍄 Step 2: Define Classifiers
st.header("🍄 2. Choose Your Classifier Power-Ups")
classifiers_input = st.text_area(
    "🎯 Enter classifier names (comma-separated):",
    "urgency_marketing, exclusive_marketing, personal_milestone, gratitude_reflection, local_business, social_proof, discount_pricing"
)
classifiers = [c.strip() for c in classifiers_input.split(",")]

# ⭐ Step 3: Process the data
st.header("⭐ 3. Process Your Adventure Data")

if uploaded_file is not None:
    try:
        decoded = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded))
    except UnicodeDecodeError:
        decoded = uploaded_file.getvalue().decode('latin1')
        df = pd.read_csv(io.StringIO(decoded))

    if st.button("🔊 Generate Statement-Level Metrics"):
        st.markdown("<script>playCoinSound()</script>", unsafe_allow_html=True)

        if "text" not in df.columns:
            st.error("Missing 'text' column in your data.")
        else:
            metrics_df = df.copy()
            for classifier in classifiers:
                col_name = f"has_{classifier}"
                if col_name in df.columns:
                    metrics_df[f"{classifier}_word_count"] = df.apply(
                        lambda row: len(str(row["text"]).split()) if row[col_name] == 1 else 0,
                        axis=1
                    )
                    metrics_df[f"{classifier}_word_pct"] = df.apply(
                        lambda row: (len(str(row["text"]).split()) if row[col_name] == 1 else 0) / max(len(str(row["text"]).split()), 1),
                        axis=1
                    )
            st.subheader("📊 Final Score: Statement Metrics")
            st.dataframe(metrics_df)

    if st.button("👑 Boss Level: Aggregate by ID"):
        st.markdown("<script>playCoinSound()</script>", unsafe_allow_html=True)

        if "ID" in df.columns:
            agg_df = df.groupby("ID")[classifiers].sum().reset_index()
            st.subheader("🏁 Aggregated Metrics by ID")
            st.dataframe(agg_df)
        else:
            st.warning("⚠️ No 'ID' column found for aggregation.")
else:
    st.info("🧭 Upload your CSV to begin the adventure!")
