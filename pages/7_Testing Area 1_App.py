import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io

# Default dictionaries for classic/timeless/luxury style
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
                # Multi-word phrase
                if keyword_lower in processed_text:
                    matched_keywords.append(keyword)
                    total_matches += 1
            else:
                # Single word with word boundaries
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
    
    # Classify all texts
    results = []
    predictions = []
    
    for idx, row in data.iterrows():
        result = classify_text(row[text_column], dictionaries)
        result['text'] = row[text_column]
        result['ground_truth'] = row[ground_truth_column]
        result['id'] = idx
        
        results.append(result)
        predictions.append(result['prediction'])
    
    # Calculate metrics
    y_true = data[ground_truth_column].values
    y_pred = np.array(predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return results, predictions, accuracy, precision, recall, f1

def create_results_dataframes(results, predictions, ground_truth):
    """Create detailed and summary dataframes"""
    
    # Detailed results
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
    
    # Summary metrics
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
        
        # Split by lines and clean up
        lines = dict_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                # Remove quotes and clean up
                keyword = line.strip('\'"').strip()
                if keyword:
                    custom_dict['custom_keywords'].add(keyword)
        
        return custom_dict if custom_dict['custom_keywords'] else None
    except Exception as e:
        st.error(f"Error parsing dictionary: {e}")
        return None

# Streamlit App
def main():
    st.set_page_config(
        page_title="Dictionary Classification Tool",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Dictionary Classification Tool")
    st.markdown("Upload your dataset and classify text using keyword dictionaries")
    
    # Sidebar for configuration
    st.sidebar.header("📊 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing your text data"
    )
    
    if uploaded_file is not None:
        # Load data with encoding handling
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
            st.success(f"✅ File loaded successfully! Shape: {data.shape}")
            
            # Column selection
            st.sidebar.subheader("📋 Column Selection")
            
            text_column = st.sidebar.selectbox(
                "Text Column",
                options=data.columns,
                index=0 if 'Statement' not in data.columns else list(data.columns).index('Statement'),
                help="Select the column containing the text to classify"
            )
            
            ground_truth_column = st.sidebar.selectbox(
                "Ground Truth Column",
                options=data.columns,
                index=1 if 'mode_researcher' not in data.columns else list(data.columns).index('mode_researcher'),
                help="Select the column containing the true labels (0 or 1)"
            )
            
            # Dictionary selection
            st.sidebar.subheader("📚 Dictionary Settings")
            
            dict_option = st.sidebar.radio(
                "Choose Dictionary Type",
                ["Default Dictionary", "Custom Dictionary"],
                help="Use the default luxury/classic dictionary or create your own"
            )
            
            if dict_option == "Default Dictionary":
                dictionaries = DEFAULT_DICTIONARIES
                
                # Show default dictionary info
                st.sidebar.info(f"Using default dictionary with {len(DEFAULT_DICTIONARIES['classic_timeless_luxury'])} keywords")
                
                with st.sidebar.expander("View Default Keywords"):
                    keywords_list = sorted(list(DEFAULT_DICTIONARIES['classic_timeless_luxury']))
                    st.write(", ".join(keywords_list))
                    
            else:
                st.sidebar.subheader("✏️ Custom Dictionary")
                
                # Pre-populate with default keywords for editing
                default_keywords = "\n".join(sorted(list(DEFAULT_DICTIONARIES['classic_timeless_luxury'])))
                
                custom_dict_text = st.sidebar.text_area(
                    "Enter Keywords (one per line)",
                    value=default_keywords,
                    height=200,
                    help="Enter keywords one per line. Lines starting with # are ignored as comments."
                )
                
                dictionaries = parse_custom_dictionary(custom_dict_text)
                
                if dictionaries is None:
                    st.sidebar.error("Invalid dictionary format. Please check your input.")
                    return
                else:
                    st.sidebar.success(f"Custom dictionary with {len(dictionaries['custom_keywords'])} keywords")
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)
                
                st.subheader("📊 Data Info")
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Total Rows", len(data))
                with info_col2:
                    st.metric("Text Column", text_column)
                with info_col3:
                    st.metric("Ground Truth Column", ground_truth_column)
            
            with col2:
                st.subheader("🎯 Ground Truth Distribution")
                gt_counts = data[ground_truth_column].value_counts()
                st.bar_chart(gt_counts)
                
                for label, count in gt_counts.items():
                    st.write(f"Label {label}: {count} ({count/len(data)*100:.1f}%)")
            
            # Run classification button
            if st.button("🚀 Run Classification", type="primary", use_container_width=True):
                
                with st.spinner("Running classification..."):
                    results, predictions, accuracy, precision, recall, f1 = run_classification(
                        data, text_column, ground_truth_column, dictionaries
                    )
                
                # Display results
                st.subheader("📊 Classification Results")
                
                # Metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with metric_col2:
                    st.metric("Precision", f"{precision:.2%}")
                with metric_col3:
                    st.metric("Recall", f"{recall:.2%}")
                with metric_col4:
                    st.metric("F1 Score", f"{f1:.2%}")
                
                # Keyword analysis
                st.subheader("🔤 Keyword Analysis")
                
                all_keywords = []
                positive_results = [r for r in results if r['prediction'] == 1]
                
                for result in positive_results:
                    all_keywords.extend(result['keywords'])
                
                keyword_counts = Counter(all_keywords)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write(f"**Texts classified as positive:** {len(positive_results)}/{len(results)}")
                    
                    if keyword_counts:
                        st.write("**Top 15 Keywords:**")
                        top_keywords = keyword_counts.most_common(15)
                        keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
                        st.dataframe(keyword_df, use_container_width=True)
                
                with col2:
                    if keyword_counts:
                        st.write("**Keyword Frequency Chart:**")
                        top_10 = dict(keyword_counts.most_common(10))
                        st.bar_chart(top_10)
                
                # Sample results
                st.subheader("📝 Sample Classifications")
                
                sample_positive = [r for r in positive_results[:5]]
                
                for i, result in enumerate(sample_positive, 1):
                    with st.expander(f"Sample {i} - GT: {result['ground_truth']}, Pred: {result['prediction']}"):
                        st.write("**Text:**", result['text'])
                        st.write("**Keywords found:**", ", ".join(result['keywords']))
                        st.write("**Total matches:**", result['matches'])
                
                # Create downloadable results
                df_detailed, df_summary = create_results_dataframes(results, predictions, data[ground_truth_column].values)
                
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Detailed results download
                    csv_detailed = df_detailed.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Detailed Results",
                        data=csv_detailed,
                        file_name="classification_detailed.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Summary results download
                    csv_summary = df_summary.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Summary Metrics",
                        data=csv_summary,
                        file_name="classification_summary.csv",
                        mime="text/csv"
                    )
                
                # Show detailed results table
                st.subheader("📋 Detailed Results")
                st.dataframe(df_detailed, use_container_width=True)
        
        else:
            st.error("❌ Could not read the file. Please check the format and encoding.")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example format
        st.subheader("📋 Expected Data Format")
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
        **Requirements:**
        - CSV file with text data and ground truth labels
        - Ground truth column should contain 0 (negative) or 1 (positive) values
        - Text column should contain the statements to classify
        """)

if __name__ == "__main__":
    main()
