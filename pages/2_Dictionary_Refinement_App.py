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
from io import StringIO

# Set page config with Toad theme
st.set_page_config(
    page_title="🍄 Toad's Mushroom Classifier",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Toad theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE4E1 50%, #FFFFFF 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #FFE4E1;
        border: 2px solid #FF6B6B;
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        background-color: #FFE4E1;
        border: 2px solid #FF6B6B;
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #FFE4E1;
        border: 2px solid #FF6B6B;
        border-radius: 10px;
    }
    
    .mushroom-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        padding: 1rem;
        border-radius: 15px;
        border: 3px solid #D63031;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(214, 48, 49, 0.3);
    }
    
    .keyword-mushroom {
        background: linear-gradient(135deg, #D63031 0%, #FF6B6B 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        display: inline-block;
        font-weight: bold;
        border: 2px solid #B71C1C;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .toad-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #D63031 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border: 3px solid #B71C1C;
        text-align: center;
    }
    
    .speed-boost {
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)

class ToadStreamlitClassifier:
    """
    🍄 Toad's Super Mushroom Streamlit Classification Tool 🍄
    """
    
    def __init__(self):
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'dictionary' not in st.session_state:
            st.session_state.dictionary = []
        if 'text_column' not in st.session_state:
            st.session_state.text_column = None
        if 'ground_truth_column' not in st.session_state:
            st.session_state.ground_truth_column = None
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'keyword_analysis' not in st.session_state:
            st.session_state.keyword_analysis = None
        if 'metrics' not in st.session_state:
            st.session_state.metrics = None
    
    def load_sample_data(self):
        """Load Toad's racing sample data"""
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
        """Auto-detect text and ground truth columns with Toad's speed"""
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
        """Perform classification with mushroom power!"""
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
        """Calculate performance metrics with Toad's precision"""
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
        """Analyze individual keyword performance with Toad's wisdom"""
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
    classifier = ToadStreamlitClassifier()
    
    # Toad Header
    st.markdown("""
    <div class="toad-header">
        <h1 style="color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🍄 TOAD'S SUPER MUSHROOM CLASSIFIER! 🍄
        </h1>
        <h3 style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
            🏁 Race through text classification with mushroom power! 🏁
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Toad's Racing Controls
    with st.sidebar:
        st.markdown("## 🏁 Toad's Racing Controls")
        
        # Data input section
        st.markdown("### 🍄 Step 1: Load Racing Data")
        
        data_option = st.radio(
            "Choose your data source:",
            ["🏁 Use Toad's Sample Data", "📁 Upload CSV File", "📝 Paste CSV Text"],
            help="Wahoo! Choose how to load your data!"
        )
        
        if data_option == "🏁 Use Toad's Sample Data":
            if st.button("🍄 Load Sample Data", help="Yahoo! Load Toad's sample racing data!"):
                st.session_state.data = classifier.load_sample_data()
                st.success("🏁 Wahoo! Sample data loaded at super speed!")
                
        elif data_option == "📁 Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv",
                help="Upload your CSV file and Toad will race through it!"
            )
            if uploaded_file is not None:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success(f"🍄 Yahoo! Loaded {len(st.session_state.data)} rows at turbo speed!")
                
        elif data_option == "📝 Paste CSV Text":
            csv_text = st.text_area(
                "Paste your CSV data here:", 
                height=150,
                help="Paste CSV data and Toad will process it with mushroom power!"
            )
            if st.button("🏁 Parse CSV") and csv_text:
                try:
                    st.session_state.data = pd.read_csv(StringIO(csv_text))
                    st.success(f"🍄 Let's-a-go! Loaded {len(st.session_state.data)} rows!")
                except Exception as e:
                    st.error(f"🚫 Mamma mia! Error parsing CSV: {e}")
        
        # Column selection
        if st.session_state.data is not None:
            st.markdown("### 🎯 Column Selection")
            
            text_col, truth_col = classifier.auto_detect_columns(st.session_state.data)
            
            text_column = st.selectbox(
                "📝 Text Column for Racing:",
                st.session_state.data.columns,
                index=st.session_state.data.columns.get_loc(text_col) if text_col else 0,
                help="Choose the column with text to classify!"
            )
            
            ground_truth_column = st.selectbox(
                "🏆 Ground Truth Column (Optional):",
                [None] + list(st.session_state.data.columns),
                index=(st.session_state.data.columns.get_loc(truth_col) + 1) if truth_col else 0,
                help="Choose the column with correct answers (0/1 values)"
            )
        
        # Dictionary input
        st.markdown("### 🍄 Step 2: Mushroom Dictionary")
        
        # Default Toad-themed dictionary
        default_keywords = "spring, trunk, show, sale, custom, price, offer, discount, limited, special"
        
        dictionary_input = st.text_area(
            "Enter your mushroom keywords (comma-separated):",
            value=default_keywords,
            height=120,
            help="Enter keywords separated by commas. Toad will use these to classify text!"
        )
        
        if st.button("🍄 Save Mushroom Dictionary", help="Save your keywords with mushroom power!"):
            keywords = [kw.strip().lower() for kw in dictionary_input.split(',') if kw.strip()]
            st.session_state.dictionary = keywords
            st.success(f"🏁 Yahoo! Dictionary saved with {len(keywords)} mushroom keywords!")
        
        # Classification button
        st.markdown("### 🏁 Step 3: Super Speed Classification")
        
        if st.button("🍄 START RACING!", type="primary", help="Let's-a-go! Begin classification!"):
            if st.session_state.data is not None and st.session_state.dictionary:
                with st.spinner("🏁 Toad is racing through your data at super speed..."):
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
                
                st.success("🏁 Wahoo! Classification complete! Toad finished the race!")
                st.balloons()
            else:
                st.error("🚫 Mamma mia! Please load data and set dictionary first!")
    
    # Main content area
    if st.session_state.data is not None:
        st.markdown("## 🔍 Racing Data Preview")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        # Show dictionary
        if st.session_state.dictionary:
            st.markdown("## 🍄 Current Mushroom Dictionary")
            keywords_html = " ".join([f'<span class="keyword-mushroom">{kw}</span>' for kw in st.session_state.dictionary])
            st.markdown(keywords_html, unsafe_allow_html=True)
    
    # Results display
    if st.session_state.results is not None:
        st.markdown("## 🏆 Toad's Super Mushroom Results!")
        
        # Metrics display
        if st.session_state.metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="mushroom-card">', unsafe_allow_html=True)
                st.metric("🎯 Accuracy", f"{st.session_state.metrics['accuracy']:.1%}", help="Overall correctness - Yahoo!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="mushroom-card">', unsafe_allow_html=True)
                st.metric("🔴 Precision", f"{st.session_state.metrics['precision']:.1%}", help="Mushroom accuracy!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="mushroom-card">', unsafe_allow_html=True)
                st.metric("⚡ Recall", f"{st.session_state.metrics['recall']:.1%}", help="Speed boost power!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="mushroom-card">', unsafe_allow_html=True)
                st.metric("🏁 F1 Score", f"{st.session_state.metrics['f1_score']:.1%}", help="Overall racing performance!")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced metrics
            st.markdown("### 🍄 Toad's Advanced Mushroom Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 MAE", f"{st.session_state.metrics['mae']:.4f}", help="Mean Absolute Error")
            with col2:
                st.metric("📐 RMSE", f"{st.session_state.metrics['rmse']:.4f}", help="Root Mean Square Error")
            with col3:
                st.metric("⭐ R²", f"{st.session_state.metrics['r2_score']:.4f}", help="R-Squared Score")
            with col4:
                st.metric("🌟 Correlation", f"{st.session_state.metrics['correlation']:.4f}", help="Correlation Coefficient")
        
        # Visualizations
        st.markdown("### 📊 Toad's Racing Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution with Toad colors
            fig = px.histogram(
                st.session_state.results, 
                x='continuous_score',
                title='🔴 Continuous Score Distribution',
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(
                plot_bgcolor='rgba(255,255,255,0.8)',
                paper_bgcolor='rgba(255,255,255,0.8)',
                font_color='#D63031'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Binary predictions pie chart
            binary_counts = st.session_state.results['binary_prediction'].value_counts()
            fig = px.pie(
                values=binary_counts.values,
                names=['Negative (0)', 'Positive (1)'],
                title='🏁 Binary Classification Results',
                color_discrete_sequence=['#74B9FF', '#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        if st.session_state.metrics:
            st.markdown("### 🎯 Mushroom Kingdom Confusion Matrix")
            
            cm_values = [
                [st.session_state.metrics['tn'], st.session_state.metrics['fp']],
                [st.session_state.metrics['fn'], st.session_state.metrics['tp']]
            ]
            
            fig = px.imshow(
                cm_values,
                text_auto=True,
                color_continuous_scale='Reds',
                title='Confusion Matrix'
            )
            fig.update_xaxes(ticktext=['Predicted 0', 'Predicted 1'], tickvals=[0, 1])
            fig.update_yaxes(ticktext=['Actual 0', 'Actual 1'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Keyword analysis
        if st.session_state.keyword_analysis is not None:
            st.markdown("### 🔍 Toad's Mushroom Keyword Analysis")
            
            # Keyword performance chart
            fig = px.bar(
                st.session_state.keyword_analysis.nlargest(10, 'f1_score'),
                x='f1_score',
                y='keyword',
                orientation='h',
                title='🍄 Top Keywords by F1 Score',
                color='f1_score',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top keywords by metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚡ Top by Recall (Speed Boost!)")
                top_recall = st.session_state.keyword_analysis.nlargest(5, 'recall')
                for _, row in top_recall.iterrows():
                    st.write(f"🍄 **{row['keyword']}** - Recall: {row['recall']:.1%}, Precision: {row['precision']:.1%}, F1: {row['f1_score']:.1%}")
            
            with col2:
                st.markdown("#### 🔴 Top by Precision (Mushroom Accuracy!)")
                top_precision = st.session_state.keyword_analysis.nlargest(5, 'precision')
                for _, row in top_precision.iterrows():
                    st.write(f"🍄 **{row['keyword']}** - Precision: {row['precision']:.1%}, Recall: {row['recall']:.1%}, F1: {row['f1_score']:.1%}")
        
        # Error analysis
        st.markdown("### 🚫 Toad's Error Analysis")
        
        if st.session_state.metrics:
            error_tab1, error_tab2 = st.tabs(["❌ False Positives (Toad's Oopsies)", "💔 False Negatives (Missed by Toad)"])
            
            with error_tab1:
                fp_results = st.session_state.results[
                    (st.session_state.results['binary_prediction'] == 1) & 
                    (st.session_state.results['ground_truth'] == 0)
                ]
                
                if len(fp_results) > 0:
                    st.markdown(f"**❌ {len(fp_results)} False Positives Found - Toad's Oopsies:**")
                    for i, (_, row) in enumerate(fp_results.head(10).iterrows()):
                        with st.expander(f"Oopsie #{i+1}"):
                            st.write(f"**Text:** {row['text']}")
                            st.write(f"**🍄 Matched Keywords:** {', '.join(row['matched_keywords'])}")
                            st.write(f"**Score:** {row['continuous_score']:.3f}")
                else:
                    st.success("🏁 Wahoo! No false positives - Perfect mushroom accuracy!")
            
            with error_tab2:
                fn_results = st.session_state.results[
                    (st.session_state.results['binary_prediction'] == 0) & 
                    (st.session_state.results['ground_truth'] == 1)
                ]
                
                if len(fn_results) > 0:
                    st.markdown(f"**💔 {len(fn_results)} False Negatives Found - Toad Missed These:**")
                    for i, (_, row) in enumerate(fn_results.head(10).iterrows()):
                        with st.expander(f"Missed #{i+1}"):
                            st.write(f"**Text:** {row['text']}")
                            st.write("**🔍 No keywords matched - Need more mushroom power!**")
                            st.write(f"**Score:** {row['continuous_score']:.3f}")
                else:
                    st.success("🍄 Yahoo! No false negatives - Perfect speed boost!")
        
        # Export section
        st.markdown("### 📥 Export Toad's Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Racing Results"):
                csv = st.session_state.results.to_csv(index=False)
                st.download_button(
                    label="🏁 Download CSV",
                    data=csv,
                    file_name="toad_classification_results.csv",
                    mime="text/csv",
                    help="Download Toad's racing results!"
                )
        
        with col2:
            if st.session_state.keyword_analysis is not None:
                if st.button("🔍 Download Keyword Analysis"):
                    csv = st.session_state.keyword_analysis.to_csv(index=False)
                    st.download_button(
                        label="🍄 Download Analysis",
                        data=csv,
                        file_name="toad_keyword_analysis.csv",
                        mime="text/csv",
                        help="Download Toad's keyword analysis!"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("*🍄 Made with mushroom power and racing spirit! Wahoo! 🏁*")

if __name__ == "__main__":
    main()
