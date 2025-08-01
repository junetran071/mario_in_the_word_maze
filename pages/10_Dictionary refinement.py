import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import io

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import base64

# Set page config
st.set_page_config(
    page_title="🌸 Princess Dictionary Classifier",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for princess theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #ffeef7 0%, #fce4ec 100%);
    }
    .stSelectbox > div > div {
        background-color: #fce4ec;
    }
    .stTextInput > div > div > input {
        background-color: #fce4ec;
    }
    .stTextArea > div > div > textarea {
        background-color: #fce4ec;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8bbd9 0%, #e1bee7 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ec407a;
        margin: 0.5rem 0;
    }
    .keyword-chip {
        background: linear-gradient(135deg, #e91e63 0%, #9c27b0 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDictionaryClassifier:
    """
    Streamlit Dictionary-based Text Classification Tool
    """
    
    def __init__(self):
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'dictionary' not in st.session_state:
            st.session_state.dictionary = []
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'keyword_analysis' not in st.session_state:
            st.session_state.keyword_analysis = None
        if 'metrics' not in st.session_state:
            st.session_state.metrics = None
    
    def load_sample_data(self):
        """Load built-in sample data"""
        sample_data = """ID,Statement,Answer
1,Its SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want and the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some wardrobe crunches and check your basics! Never on sale.,1
5,He is a hard worker and always willing to lend a hand. The prices are the best I have seen in 17 years of servicing my clients.,0
6,Check out our EXCLUSIVE summer collection with amazing DISCOUNTS!,1
7,The weather is nice today and I hope you have a great day.,0
8,LIMITED TIME OFFER on all designer items - dont miss out!,1
9,Thank you for your patience while we process your request.,0
10,NEW ARRIVALS are here with SPECIAL PRICING just for you!,1"""
        
        st.session_state.data = pd.read_csv(StringIO(sample_data))
        return st.session_state.data
    
    def auto_detect_columns(self, data):
        """Auto-detect text and ground truth columns"""
        columns = data.columns.tolist()
        
        text_column = None
        ground_truth_column = None
        
        # Detect text column
        if 'Statement' in columns:
            text_column = 'Statement'
        elif 'statement' in columns:
            text_column = 'statement'
        elif 'text' in columns:
            text_column = 'text'
        else:
            # Use first text-heavy column
            for col in columns:
                if data[col].dtype == 'object':
                    avg_length = data[col].astype(str).str.len().mean()
                    if avg_length > 20:
                        text_column = col
                        break
        
        # Detect ground truth column
        if 'Answer' in columns:
            ground_truth_column = 'Answer'
        elif 'answer' in columns:
            ground_truth_column = 'answer'
        elif 'label' in columns:
            ground_truth_column = 'label'
        elif 'target' in columns:
            ground_truth_column = 'target'
            
        return text_column, ground_truth_column
    
    def classify_text(self, data, dictionary, text_column, ground_truth_column=None):
        """Perform classification using the dictionary"""
        results = []
        
        for idx, row in data.iterrows():
            text = str(row[text_column]).lower()
            
            # Find matched keywords
            matched_keywords = []
            keyword_frequencies = []
            
            for keyword in dictionary:
                if keyword in text:
                    matched_keywords.append(keyword)
                    freq = len(re.findall(re.escape(keyword), text))
                    keyword_frequencies.append(freq)
                else:
                    keyword_frequencies.append(0)
            
            # Binary prediction
            binary_prediction = 1 if matched_keywords else 0
            
            # Continuous scores
            continuous_score = len(matched_keywords) / len(dictionary) if dictionary else 0
            frequency_score = sum(keyword_frequencies)
            
            # Ground truth if available
            ground_truth = None
            if ground_truth_column and ground_truth_column in data.columns:
                try:
                    ground_truth = int(row[ground_truth_column])
                except:
                    ground_truth = None
            
            results.append({
                'text': row[text_column],
                'binary_prediction': binary_prediction,
                'continuous_score': continuous_score,
                'frequency_score': frequency_score,
                'matched_keywords': matched_keywords,
                'ground_truth': ground_truth
            })
        
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df):
        """Calculate performance metrics"""
        valid_results = results_df[results_df['ground_truth'].notna()].copy()
        
        if len(valid_results) == 0:
            return None
            
        y_true = valid_results['ground_truth'].values
        y_pred_binary = valid_results['binary_prediction'].values
        y_pred_continuous = valid_results['continuous_score'].values
        
        # Binary classification metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Continuous metrics
        mae = mean_absolute_error(y_true, y_pred_continuous)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_continuous))
        r2 = r2_score(y_true, y_pred_continuous)
        correlation, _ = pearsonr(y_true, y_pred_continuous)
        
        # Confusion matrix components
        tp = len(valid_results[(valid_results['binary_prediction'] == 1) & (valid_results['ground_truth'] == 1)])
        fp = len(valid_results[(valid_results['binary_prediction'] == 1) & (valid_results['ground_truth'] == 0)])
        tn = len(valid_results[(valid_results['binary_prediction'] == 0) & (valid_results['ground_truth'] == 0)])
        fn = len(valid_results[(valid_results['binary_prediction'] == 0) & (valid_results['ground_truth'] == 1)])
        
        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
            'mae': mae, 'rmse': rmse, 'r2_score': r2, 'correlation': correlation,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    def analyze_keywords(self, data, results_df, dictionary, text_column, ground_truth_column):
        """Analyze individual keyword performance"""
        valid_results = results_df[results_df['ground_truth'].notna()].copy()
        keyword_metrics = []
        
        for keyword in dictionary:
            # Find statements where this keyword appears
            keyword_present = valid_results.apply(
                lambda row: keyword in row['matched_keywords'], axis=1
            )
            
            # Calculate metrics for this keyword
            true_positives = len(valid_results[keyword_present & (valid_results['ground_truth'] == 1)])
            false_positives = len(valid_results[keyword_present & (valid_results['ground_truth'] == 0)])
            total_positives = len(valid_results[valid_results['ground_truth'] == 1])
            
            recall = true_positives / total_positives if total_positives > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Get examples
            tp_examples = valid_results[keyword_present & (valid_results['ground_truth'] == 1)]['text'].head(3).tolist()
            fp_examples = valid_results[keyword_present & (valid_results['ground_truth'] == 0)]['text'].head(3).tolist()
            
            keyword_metrics.append({
                'keyword': keyword,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'tp_examples': tp_examples,
                'fp_examples': fp_examples
            })
        
        return pd.DataFrame(keyword_metrics)

def main():
    classifier = StreamlitDictionaryClassifier()
    
    # Header
    st.markdown("# 🌸 Princess Dictionary Classification Bot 👑")
    st.markdown("*Enter keywords and classify statements to analyze their royal effectiveness*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📚 Royal Controls")
        
        # Data input section
        st.markdown("### 📊 Step 1: Load Royal Data")
        
        data_option = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload CSV File", "Paste CSV Text"]
        )
        
        if data_option == "Use Sample Data":
            if st.button("🎀 Load Sample Data"):
                st.session_state.data = classifier.load_sample_data()
                st.success("✨ Sample data loaded!")
                
        elif data_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success(f"📥 Loaded {len(st.session_state.data)} rows!")
                
        elif data_option == "Paste CSV Text":
            csv_text = st.text_area("Paste your CSV data here:", height=150)
            if st.button("📝 Parse CSV") and csv_text:
                try:
                    st.session_state.data = pd.read_csv(StringIO(csv_text))
                    st.success(f"📥 Loaded {len(st.session_state.data)} rows!")
                except Exception as e:
                    st.error(f"❌ Error parsing CSV: {e}")
        
        # Column selection
        if st.session_state.data is not None:
            st.markdown("### 🎯 Column Selection")
            
            text_col, truth_col = classifier.auto_detect_columns(st.session_state.data)
            
            text_column = st.selectbox(
                "📝 Text Column:",
                st.session_state.data.columns,
                index=st.session_state.data.columns.get_loc(text_col) if text_col else 0
            )
            
            ground_truth_column = st.selectbox(
                "🎯 Ground Truth Column (Optional):",
                [None] + list(st.session_state.data.columns),
                index=(st.session_state.data.columns.get_loc(truth_col) + 1) if truth_col else 0
            )
        
        # Dictionary input
        st.markdown("### 📚 Step 2: Royal Dictionary")
        
        dictionary_input = st.text_area(
            "Enter keywords (comma-separated):",
            value="spring, trunk, show, sale, custom, price, offer, discount",
            height=100
        )
        
        if st.button("💾 Save Dictionary"):
            keywords = [kw.strip().lower() for kw in dictionary_input.split(',') if kw.strip()]
            st.session_state.dictionary = keywords
            st.success(f"✨ Dictionary saved with {len(keywords)} keywords!")
        
        # Classification button
        st.markdown("### 🎭 Step 3: Royal Classification")
        
        if st.button("👸 Begin Classification", type="primary"):
            if st.session_state.data is not None and st.session_state.dictionary:
                with st.spinner("🔮 Performing royal classification..."):
                    st.session_state.results = classifier.classify_text(
                        st.session_state.data, 
                        st.session_state.dictionary, 
                        text_column, 
                        ground_truth_column
                    )
                    
                    if ground_truth_column:
                        st.session_state.metrics = classifier.calculate_metrics(st.session_state.results)
                        st.session_state.keyword_analysis = classifier.analyze_keywords(
                            st.session_state.data, 
                            st.session_state.results, 
                            st.session_state.dictionary, 
                            text_column, 
                            ground_truth_column
                        )
                
                st.success("🎉 Classification complete!")
            else:
                st.error("❌ Please load data and set dictionary first!")
    
    # Main content area
    if st.session_state.data is not None:
        st.markdown("## 🔍 Data Preview")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        # Show dictionary
        if st.session_state.dictionary:
            st.markdown("## ✨ Current Dictionary")
            keywords_html = " ".join([f'<span class="keyword-chip">{kw}</span>' for kw in st.session_state.dictionary])
            st.markdown(keywords_html, unsafe_allow_html=True)
    
    # Results display
    if st.session_state.results is not None:
        st.markdown("## 👑 Royal Classification Results")
        
        # Metrics display
        if st.session_state.metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Accuracy", f"{st.session_state.metrics['accuracy']:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("💎 Precision", f"{st.session_state.metrics['precision']:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🌟 Recall", f"{st.session_state.metrics['recall']:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("👸 F1 Score", f"{st.session_state.metrics['f1_score']:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced metrics
            st.markdown("### 🔮 Advanced Royal Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 MAE", f"{st.session_state.metrics['mae']:.4f}")
            with col2:
                st.metric("📐 RMSE", f"{st.session_state.metrics['rmse']:.4f}")
            with col3:
                st.metric("💫 R²", f"{st.session_state.metrics['r2_score']:.4f}")
            with col4:
                st.metric("🌊 Correlation", f"{st.session_state.metrics['correlation']:.4f}")
        
        # Visualizations
        st.markdown("### 📊 Royal Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig = px.histogram(
                st.session_state.results, 
                x='continuous_score',
                title='🌸 Continuous Score Distribution',
                color_discrete_sequence=['#e91e63']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Binary predictions pie chart
            binary_counts = st.session_state.results['binary_prediction'].value_counts()
            fig = px.pie(
                values=binary_counts.values,
                names=['Negative (0)', 'Positive (1)'],
                title='👸 Binary Classification Results',
                color_discrete_sequence=['#81c784', '#f06292']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        if st.session_state.metrics:
            st.markdown("### 🎯 Confusion Matrix")
            
            cm_data = [
                ['True Negative', 'False Positive'],
                ['False Negative', 'True Positive']
            ]
            cm_values = [
                [st.session_state.metrics['tn'], st.session_state.metrics['fp']],
                [st.session_state.metrics['fn'], st.session_state.metrics['tp']]
            ]
            
            fig = px.imshow(
                cm_values,
                text_auto=True,
                color_continuous_scale='RdYlBu_r',
                title='Confusion Matrix'
            )
            fig.update_xaxes(ticktext=['Predicted 0', 'Predicted 1'], tickvals=[0, 1])
            fig.update_yaxes(ticktext=['Actual 0', 'Actual 1'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Keyword analysis
        if st.session_state.keyword_analysis is not None:
            st.markdown("### 🔍 Royal Keyword Analysis")
            
            # Keyword performance chart
            fig = px.bar(
                st.session_state.keyword_analysis.nlargest(10, 'f1_score'),
                x='f1_score',
                y='keyword',
                orientation='h',
                title='✨ Top Keywords by F1 Score',
                color='f1_score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed keyword table
            st.markdown("#### 📈 Detailed Keyword Metrics")
            display_df = st.session_state.keyword_analysis[['keyword', 'recall', 'precision', 'f1_score', 'true_positives', 'false_positives']].copy()
            display_df['recall'] = display_df['recall'].apply(lambda x: f"{x:.1%}")
            display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.1%}")
            display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True)
        
        # Error analysis
        st.markdown("### 🚫 Error Analysis")
        
        error_tab1, error_tab2 = st.tabs(["False Positives", "False Negatives"])
        
        with error_tab1:
            if st.session_state.metrics:
                fp_results = st.session_state.results[
                    (st.session_state.results['binary_prediction'] == 1) & 
                    (st.session_state.results['ground_truth'] == 0)
                ]
                
                if len(fp_results) > 0:
                    st.markdown(f"**🚫 {len(fp_results)} False Positives Found:**")
                    for i, (_, row) in enumerate(fp_results.head(10).iterrows()):
                        with st.expander(f"False Positive {i+1}"):
                            st.write(f"**Text:** {row['text']}")
                            st.write(f"**Matched Keywords:** {', '.join(row['matched_keywords'])}")
                            st.write(f"**Score:** {row['continuous_score']:.3f}")
                else:
                    st.success("✨ No false positives - Perfect princess precision!")
        
        with error_tab2:
            if st.session_state.metrics:
                fn_results = st.session_state.results[
                    (st.session_state.results['binary_prediction'] == 0) & 
                    (st.session_state.results['ground_truth'] == 1)
                ]
                
                if len(fn_results) > 0:
                    st.markdown(f"**💔 {len(fn_results)} False Negatives Found:**")
                    for i, (_, row) in enumerate(fn_results.head(10).iterrows()):
                        with st.expander(f"False Negative {i+1}"):
                            st.write(f"**Text:** {row['text']}")
                            st.write("**Matched Keywords:** None")
                            st.write(f"**Score:** {row['continuous_score']:.3f}")
                else:
                    st.success("✨ No false negatives - Royal recall perfection!")
        
        # Export section
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Results CSV"):
                csv = st.session_state.results.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="royal_classification_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.session_state.keyword_analysis is not None:
                if st.button("🔍 Download Keyword Analysis"):
                    csv = st.session_state.keyword_analysis.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis",
                        data=csv,
                        file_name="royal_keyword_analysis.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("*🌸 Made with royal magic and Streamlit 👑*")

if __name__ == "__main__":
    main()
