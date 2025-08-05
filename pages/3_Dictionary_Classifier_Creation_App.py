import streamlit as st
import pandas as pd
import re
from collections import defaultdict

# Mario theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #fceabb, #f8b500);
        font-family: 'Press Start 2P', cursive;
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
    }
    .stDownloadButton button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🍄 Mario Text Classifier 🍄</h1>', unsafe_allow_html=True)


def create_classification_dictionary():
    return {
        "product_quality": ["excellent", "perfect", "classic", "gorgeous", "quality", "premium", "superior", "outstanding"],
        "urgency": ["last week", "limited time", "hurry", "deadline", "expires", "final", "closing", "urgent"],
        "personal_connection": ["smile", "goal", "personal", "relationship", "connect", "touch", "care", "understand"],
        "travel_business": ["travel", "suit", "business", "professional", "work", "office", "corporate", "formal"],
        "seasonal_holiday": ["holidays", "christmas", "season", "winter", "summer", "spring", "fall", "celebration"]
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


st.write("Upload your CSV file and classify text using keyword dictionaries")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
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
                df[f'{category}_match'] = df['classifications'].apply(
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
            display_df = df[[text_column] + [f'{cat}_match' for cat in classification_dict.keys()]].copy()

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

