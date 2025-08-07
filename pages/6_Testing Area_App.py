import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io

# Page config with Bowser theme
st.set_page_config(
    page_title="👑 Bowser's Classification Castle",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bowser Mario style with white background
st.markdown("""
<style>
    /* Main background */
    .main > div {
        background-color: white;
        padding: 2rem;
    }
    
    /* Headers with Bowser colors */
    h1 {
        color: #8B4513 !important;
        font-family: 'Arial Black', sans-serif !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        border-bottom: 3px solid #FF6B35;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #CD5C5C !important;
        font-weight: bold !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFF8DC !important;
        border-right: 3px solid #8B4513;
    }
    
    /* Buttons with Bowser theme */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B35, #8B4513) !important;
        color: white !important;
        font-weight: bold !important;
        border: 2px solid #654321 !important;
        border-radius: 10px !important;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 5px 5px 12px rgba(0,0,0,0.4) !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #FFF8DC, #F5DEB3) !important;
        border: 2px solid #D2691E !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2) !important;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: #90EE90 !important;
        border: 2px solid #228B22 !important;
        border-radius: 10px !important;
    }
    
    .stInfo {
        background-color: #E6F3FF !important;
        border: 2px solid #4169E1 !important;
        border-radius: 10px !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #FFF8DC !important;
        border: 2px dashed #8B4513 !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 2px solid #D2691E !important;
        border-radius: 10px !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F5DEB3 !important;
        border: 1px solid #8B4513 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Default dictionaries
DEFAULT_DICTIONARIES = {
    'classic_timeless_luxury': {
        'timeless', 'heritage', 'legacy', 'traditional', 'enduring', 'eternal', 
        'sophisticated', 'elegant', 'refined', 'prestigious', 'distinguished',
        'exquisite', 'quality', 'craftsmanship', 'artistry', 'excellence', 
        'perfection', 'flawless', 'immaculate', 'luxury', 'premium', 'exclusive',
        'vintage', 'antique', 'heirloom', 'treasured', 'prized', 'coveted',
        'rare', 'precious', 'priceless', 'invaluable', 'unique', 'extraordinary',
        'remarkable', 'outstanding', 'exceptional', 'special', 'distinctive',
        'signature', 'original', 'authentic', 'genuine', 'iconic', 'superior',
        'ultimate', 'quintessential', 'definitive', 'benchmark', 'standard',
        'exemplar', 'world-class', 'world class', 'magnificent', 'splendid',
        'valuable', 'investment', 'grand', 'opulent', 'sumptuous', 'lavish'
    }
}

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def classify_text(text, dictionaries):
    """Classify a single text using dictionaries"""
    processed_text = preprocess_text(text)
    
    total_matches = 0
    matched_keywords = []
    
    for category, keywords in dictionaries.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            if ' ' in keyword_lower:
                if keyword_lower in processed_text:
                    matched_keywords.append(keyword)
                    total_matches += 1
            else:
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, processed_text):
                    matched_keywords.append(keyword)
                    total_matches += 1
    
    return {
        'prediction': 1 if total_matches > 0 else 0,
        'matches': total_matches,
        'keywords': matched_keywords
    }

def run_classification(data, text_column, ground_truth_column, dictionaries):
    """Run classification analysis"""
    results = []
    predictions = []
    
    for idx, row in data.iterrows():
        result = classify_text(row[text_column], dictionaries)
        result['text'] = row[text_column]
        result['ground_truth'] = row[ground_truth_column]
        result['id'] = idx
        
        results.append(result)
        predictions.append(result['prediction'])
    
    y_true = data[ground_truth_column].values
    y_pred = np.array(predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return results, predictions, accuracy, precision, recall, f1

def create_results_dataframes(results, predictions, ground_truth):
    """Create detailed and summary dataframes"""
    df_detailed = pd.DataFrame([
        {
            'id': r['id'],
            'text': r['text'],
            'ground_truth': r['ground_truth'],
            'prediction': r['prediction'],
            'keyword_matches': r['matches'],
            'keywords_found': ", ".join(r['keywords'])
        }
        for r in results
    ])
    
    y_true = ground_truth
    y_pred = np.array(predictions)
    
    df_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
        'Score': [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        ]
    })
    
    return df_detailed, df_summary

def parse_custom_dictionary(dict_text):
    """Parse custom dictionary from text input"""
    try:
        custom_dict = {'custom_keywords': set()}
        
        lines = dict_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                keyword = line.strip('\'"').strip()
                if keyword:
                    custom_dict['custom_keywords'].add(keyword)
        
        return custom_dict if custom_dict['custom_keywords'] else None
    except Exception as e:
        st.error(f"🔥 Bowser says: Error parsing dictionary - {e}")
        return None

def main():
    # Title with Bowser theme
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>👑 Bowser's Classification Castle 🔥</h1>
        <p style="font-size: 18px; color: #8B4513; font-weight: bold;">
            Conquer text classification with the power of King Koopa! 🐢
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🏰 Castle Controls")
        
        uploaded_file = st.file_uploader(
            "📜 Upload Your Scroll (CSV)",
            type=['csv'],
            help="Bowser demands a proper CSV file!"
        )
    
    if uploaded_file is not None:
        @st.cache_data
        def load_data(file):
            encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    return pd.read_csv(file, encoding=encoding)
                except:
                    continue
            return None
        
        data = load_data(uploaded_file)
        
        if data is not None:
            st.success(f"🔥 Excellent! Data loaded successfully! Shape: {data.shape}")
            
            # Sidebar controls
            with st.sidebar:
                st.markdown("### ⚙️ Battle Configuration")
                
                text_column = st.selectbox(
                    "📝 Text Column",
                    options=data.columns,
                    index=0 if 'Statement' not in data.columns else list(data.columns).index('Statement')
                )
                
                ground_truth_column = st.selectbox(
                    "🎯 Truth Column",
                    options=data.columns,
                    index=1 if 'mode_researcher' not in data.columns else list(data.columns).index('mode_researcher')
                )
                
                st.markdown("### 📚 Dictionary Arsenal")
                
                dict_option = st.radio(
                    "Choose Your Weapon",
                    ["🛡️ Default Dictionary", "⚔️ Custom Dictionary"]
                )
                
                if dict_option == "🛡️ Default Dictionary":
                    dictionaries = DEFAULT_DICTIONARIES
                    st.info(f"🔥 {len(DEFAULT_DICTIONARIES['classic_timeless_luxury'])} keywords ready for battle!")
                    
                    with st.expander("👀 View Arsenal"):
                        keywords_list = sorted(list(DEFAULT_DICTIONARIES['classic_timeless_luxury']))
                        st.write(", ".join(keywords_list))
                        
                else:
                    default_keywords = "\n".join(sorted(list(DEFAULT_DICTIONARIES['classic_timeless_luxury'])))
                    
                    custom_dict_text = st.text_area(
                        "✏️ Forge Your Keywords",
                        value=default_keywords,
                        height=200,
                        help="One keyword per line. Lines with # are ignored."
                    )
                    
                    dictionaries = parse_custom_dictionary(custom_dict_text)
                    
                    if dictionaries is None:
                        st.error("💥 Invalid format! Bowser is not pleased!")
                        return
                    else:
                        st.success(f"⚔️ Custom arsenal: {len(dictionaries['custom_keywords'])} keywords")
            
            # Main content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📊 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Battle Stats")
                gt_counts = data[ground_truth_column].value_counts()
                
                st.metric("Total Records", len(data))
                st.metric("Text Field", text_column)
                st.metric("Truth Field", ground_truth_column)
                
                st.bar_chart(gt_counts)
                
                for label, count in gt_counts.items():
                    st.write(f"**Label {label}:** {count} ({count/len(data)*100:.1f}%)")
            
            # Classification button
            st.markdown("---")
            if st.button("🔥 UNLEASH BOWSER'S POWER! 🔥", type="primary", use_container_width=True):
                
                with st.spinner("🐢 Bowser is analyzing your data..."):
                    results, predictions, accuracy, precision, recall, f1 = run_classification(
                        data, text_column, ground_truth_column, dictionaries
                    )
                
                st.markdown("### 🏆 Battle Results")
                
                # Metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("🔍 Precision", f"{precision:.2%}")
                with col3:
                    st.metric("📊 Recall", f"{recall:.2%}")
                with col4:
                    st.metric("⚖️ F1 Score", f"{f1:.2%}")
                
                # Keyword analysis
                st.markdown("### 🔤 Keyword Conquest")
                
                all_keywords = []
                positive_results = [r for r in results if r['prediction'] == 1]
                
                for result in positive_results:
                    all_keywords.extend(result['keywords'])
                
                keyword_counts = Counter(all_keywords)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**🔥 Victories:** {len(positive_results)}/{len(results)}")
                    
                    if keyword_counts:
                        st.write("**👑 Top 15 Keywords:**")
                        top_keywords = keyword_counts.most_common(15)
                        keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
                        st.dataframe(keyword_df, use_container_width=True)
                
                with col2:
                    if keyword_counts:
                        st.write("**📊 Power Chart:**")
                        top_10 = dict(keyword_counts.most_common(10))
                        st.bar_chart(top_10)
                
                # Sample results
                st.markdown("### 📝 Battle Examples")
                
                sample_positive = positive_results[:5]
                
                for i, result in enumerate(sample_positive, 1):
                    with st.expander(f"🔥 Victory {i} - Truth: {result['ground_truth']}, Prediction: {result['prediction']}"):
                        st.write("**Text:**", result['text'])
                        st.write("**Keywords:**", ", ".join(result['keywords']))
                        st.write("**Matches:**", result['matches'])
                
                # Download results
                df_detailed, df_summary = create_results_dataframes(results, predictions, data[ground_truth_column].values)
                
                st.markdown("### 💾 Claim Your Treasures")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_detailed = df_detailed.to_csv(index=False)
                    st.download_button(
                        label="📜 Download Detailed Results",
                        data=csv_detailed,
                        file_name="bowser_classification_detailed.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_summary = df_summary.to_csv(index=False)
                    st.download_button(
                        label="🏆 Download Summary",
                        data=csv_summary,
                        file_name="bowser_classification_summary.csv",
                        mime="text/csv"
                    )
                
                st.markdown("### 📋 Full Battle Report")
                st.dataframe(df_detailed, use_container_width=True)
        
        else:
            st.error("💥 Bowser cannot read this file! Check your format!")
    
    else:
        st.markdown("### 🏰 Welcome to Bowser's Castle!")
        st.info("👆 Upload a CSV file to begin your classification quest!")
        
        st.markdown("### 📋 Required Format")
        example_data = pd.DataFrame({
            'Statement': [
                'This luxury watch represents timeless elegance and sophisticated craftsmanship.',
                'A simple everyday item for basic use.',
                'Heritage quality with exceptional attention to detail and premium materials.'
            ],
            'mode_researcher': [1, 0, 1]
        })
        st.dataframe(example_data, use_container_width=True)
        
        st.markdown("""
        **🔥 Bowser's Requirements:**
        - CSV file with text and truth labels
        - Truth column: 0 (negative) or 1 (positive)
        - Text column: statements to classify
        """)

if __name__ == "__main__":
    main()
