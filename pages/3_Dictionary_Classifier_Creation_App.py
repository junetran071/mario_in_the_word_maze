import streamlit as st
import pandas as pd
import re
from collections import defaultdict

# Mario theme CSS with white background
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    .stApp {
        background-color: white !important;
        font-family: 'Press Start 2P', cursive;
        color: #000;
    }

    .main-header {
        color: #ff0000;
        text-align: center;
        font-size: 2rem;
        text-shadow: 2px 2px #000;
        margin-bottom: 20px;
    }

    .stButton > button {
        background-color: #ff0000;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: 3px solid yellow;
        box-shadow: 2px 2px #000;
        font-family: 'Press Start 2P', cursive;
    }

    .stDownloadButton button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 8px 16px;
        border: 2px solid #1e7e34;
        box-shadow: 2px 2px #000;
        font-family: 'Press Start 2P', cursive;
    }

    .stTextInput input, .stSelectbox select, .stFileUploader, .stTextArea textarea {
        border: 2px solid #ffd60a !important;
        background-color: #fff3cd !important;
        color: #000 !important;
        font-weight: bold !important;
        font-family: 'Press Start 2P', cursive !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎄 Mario Text Classifier 🎄</h1>', unsafe_allow_html=True)

# Classification Dictionary
def create_classification_dictionary():
    return {
        "urgency_marketing": ["now", "today", "limited time", "act fast", "hurry", "urgent", "last chance"],
        "exclusive_marketing": ["exclusive", "members only", "invite only", "vip", "limited access"],
        "personal_milestone": ["birthday", "anniversary", "personal win", "milestone", "special moment"],
        "gratitude_reflection": ["thank you", "grateful", "appreciate", "reflecting", "gratitude"],
        "local_business": ["local", "community", "neighborhood", "support small", "near you"],
        "social_proof": ["customers love", "top rated", "testimonials", "as seen on", "popular choice"],
        "discount_pricing": ["sale", "discount", "% off", "bogo", "markdown", "deal"]
    }

def classify_text(text, dictionary):
    if not text or pd.isna(text):
        return {}

    text_lower = text.lower()
    results = {}

    for category, keywords in dictionary.items():
        matches = [keyword for keyword in keywords if keyword.lower() in text_lower]
        if matches:
            results[category] = {
                'matched_keywords': matches,
                'count': len(matches)
            }

    return results

def classify_dataframe(df, text_column, dictionary):
    classifications = []
    for _, row in df.iterrows():
        text = row[text_column]
        classification = classify_text(text, dictionary)
        classifications.append(classification)
    return classifications

# UI
st.write("Upload your CSV file and classify text using keyword dictionaries")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
    except Exception:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    st.subheader("Data Preview")
    st.dataframe(df.head())

    text_column = st.selectbox("Select text column to classify:", df.columns)

    st.subheader("Classification Dictionary")
    classification_dict = create_classification_dictionary()

    for category, keywords in classification_dict.items():
        st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(keywords)}")

    if st.button("Classify Text"):
        with st.spinner("Classifying..."):
            classifications = classify_dataframe(df, text_column, classification_dict)
            df['classifications'] = classifications

            for category in classification_dict.keys():
                df[category] = df['classifications'].apply(
                    lambda x: x.get(category, {}).get('count', 0)
                )

            st.subheader("Classification Results")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Classification Summary:**")
                total_texts = len(df)
                for category in classification_dict.keys():
                    matches = sum(1 for c in classifications if category in c)
                    percentage = (matches / total_texts) * 100
                    st.write(f"{category.replace('_', ' ').title()}: {matches}/{total_texts} ({percentage:.1f}%)")

            with col2:
                st.write("**Most Common Categories:**")
                category_counts = defaultdict(int)
                for classification in classifications:
                    for category in classification.keys():
                        category_counts[category] += 1
                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                for category, count in sorted_categories[:5]:
                    st.write(f"{category.replace('_', ' ').title()}: {count}")

            st.subheader("Detailed Results")
            display_df = df[[text_column] + list(classification_dict.keys())].copy()

            def format_classification(row):
                details = []
                classification = df.loc[row.name, 'classifications']
                for category, info in classification.items():
                    keywords = ', '.join(info['matched_keywords'])
                    details.append(f"{category}: {keywords}")
                return '; '.join(details) if details else "No matches"

            display_df['matched_categories'] = display_df.apply(format_classification, axis=1)
            st.dataframe(display_df)

            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Classification Results",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )
else:
    st.info("Upload a CSV file to begin classification.")
