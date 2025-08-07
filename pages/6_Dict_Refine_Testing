import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import json
import io

class DictionaryClassificationBot:
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        """Reset all state variables"""
        self.tactic = ""
        self.data = None
        self.text_column = None
        self.ground_truth_column = None
        self.custom_prompt = "Generate a list of single-word (unigram) keywords for a text classification dictionary focused on the 'tactic' based on the 'context'"
        self.dictionary = []
        self.results = []
        self.keyword_analysis = {}
        
        # Sample data
        self.sample_data = pd.DataFrame([
            {"id": 1, "Statement": "This timeless piece features classic design elements", "mode_researcher": 1},
            {"id": 2, "Statement": "Limited time offer - 50% off everything!", "mode_researcher": 0},
            {"id": 3, "Statement": "Luxury craftsmanship meets modern sophistication", "mode_researcher": 1},
            {"id": 4, "Statement": "Free shipping on all orders today only", "mode_researcher": 0},
            {"id": 5, "Statement": "Heritage quality that endures through generations", "mode_researcher": 1},
            {"id": 6, "Statement": "World class cars meet world class clothing", "mode_researcher": 1},
            {"id": 7, "Statement": "Sale ends tonight - don't miss out!", "mode_researcher": 0},
            {"id": 8, "Statement": "Exquisite attention to detail in every stitch", "mode_researcher": 1}
        ])

    def generate_ai_dictionary(self, context, tactic, custom_prompt):
        """Generate dictionary using Claude AI"""
        prompt = custom_prompt.replace("'tactic'", f'"{tactic}"').replace("'context'", f'"{context}"')
        
        full_prompt = f"""{prompt}

Please provide only a comma-separated list of single words, formatted like: word1,word2,word3

Focus on unigrams (single words) that would be effective for identifying "{tactic}" in text.
Return only the keywords, no explanations."""
        
        try:
            # Claude API call
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": full_prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                keywords_text = data['content'][0]['text']
                
                # Parse keywords
                keywords = [word.strip().strip('"\'') for word in keywords_text.split(',')]
                keywords = [word for word in keywords if word and word.isalpha()]
                
                return keywords
            else:
                st.error(f"AI generation failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"AI generation error: {e}")
            return None

    def classify_texts(self, data, text_column, ground_truth_column, dictionary):
        """Run classification on the data"""
        results = []
        predictions = []
        
        for idx, row in data.iterrows():
            text = str(row[text_column]).lower()
            matched_keywords = [kw for kw in dictionary if kw.lower() in text]
            
            prediction = 1 if matched_keywords else 0
            ground_truth = int(row[ground_truth_column])
            
            result = {
                'id': idx,
                'text': row[text_column],
                'prediction': prediction,
                'ground_truth': ground_truth,
                'matched_keywords': matched_keywords,
                'keyword_count': len(matched_keywords)
            }
            
            results.append(result)
            predictions.append(prediction)
        
        return results, predictions

    def calculate_metrics(self, results):
        """Calculate classification metrics"""
        y_true = [r['ground_truth'] for r in results]
        y_pred = [r['prediction'] for r in results]
        
        return {
            'accuracy': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, zero_division=0) * 100,
            'recall': recall_score(y_true, y_pred, zero_division=0) * 100,
            'f1': f1_score(y_true, y_pred, zero_division=0) * 100
        }

    def analyze_keywords(self, results, dictionary):
        """Analyze keyword performance"""
        analysis = {}
        
        for keyword in dictionary:
            # Find results containing this keyword
            keyword_results = [r for r in results if keyword.lower() in r['text'].lower()]
            
            if not keyword_results:
                continue
            
            # Calculate metrics
            tp = len([r for r in keyword_results if r['ground_truth'] == 1 and r['prediction'] == 1])
            fp = len([r for r in keyword_results if r['ground_truth'] == 0 and r['prediction'] == 1])
            fn_all = [r for r in results if r['ground_truth'] == 1 and keyword.lower() not in r['text'].lower()]
            fn = len(fn_all)
            
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            analysis[keyword] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': [r for r in keyword_results if r['ground_truth'] == 1 and r['prediction'] == 1],
                'false_positives': [r for r in keyword_results if r['ground_truth'] == 0 and r['prediction'] == 1],
                'total_matches': len(keyword_results)
            }
        
        return analysis

def main():
    st.set_page_config(
        page_title="Dictionary Classification Bot",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎯 Dictionary Classification Bot")
    st.markdown("Create unigram dictionaries using Claude AI for text classification")

    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = DictionaryClassificationBot()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    steps = [
        "1. Define Tactic",
        "2. Upload Data",
        "3. Configure Columns",
        "4. Customize Prompt",
        "5. Generate Dictionary",
        "6. Run Classification",
        "7. View Results"
    ]
    
    current_step = st.sidebar.radio("Select Step:", steps)

    # Main content area
    if current_step == "1. Define Tactic":
        st.header("📝 Step 1: Define Your Classification Tactic")
        
        tactic = st.text_input(
            "Enter your classification tactic:",
            value=st.session_state.get('tactic', ''),
            placeholder="e.g., classic_timeless_luxury_style"
        )
        
        if st.button("Save Tactic"):
            if tactic.strip():
                st.session_state.tactic = tactic.strip()
                st.session_state.bot.tactic = tactic.strip()
                st.success(f"✅ Tactic saved: {tactic}")
            else:
                st.error("❌ Please enter a tactic")

    elif current_step == "2. Upload Data":
        st.header("📁 Step 2: Upload Your Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with text data and ground truth labels"
            )
            
            if uploaded_file is not None:
                try:
                    # Try different encodings
                    for encoding in ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']:
                        try:
                            data = pd.read_csv(uploaded_file, encoding=encoding)
                            st.session_state.data = data
                            st.session_state.bot.data = data
                            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
                            st.dataframe(data.head())
                            break
                        except Exception:
                            continue
                    else:
                        st.error("❌ Could not read the file. Please check the format.")
                        
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with col2:
            st.subheader("Use Sample Data")
            if st.button("Load Sample Data"):
                st.session_state.data = st.session_state.bot.sample_data.copy()
                st.session_state.bot.data = st.session_state.bot.sample_data.copy()
                st.success("✅ Sample data loaded!")
                st.dataframe(st.session_state.bot.sample_data)

    elif current_step == "3. Configure Columns":
        st.header("📋 Step 3: Select Text and Ground Truth Columns")
        
        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first (Step 2)")
        else:
            data = st.session_state.data
            columns = list(data.columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Text Column")
                text_candidates = [col for col in columns 
                                 if any(word in col.lower() for word in ['statement', 'text', 'content'])]
                
                default_text = text_candidates[0] if text_candidates else columns[0]
                
                text_column = st.selectbox(
                    "Select column containing text data:",
                    columns,
                    index=columns.index(default_text) if default_text in columns else 0
                )
                
                st.session_state.text_column = text_column
                st.session_state.bot.text_column = text_column
            
            with col2:
                st.subheader("Ground Truth Column")
                truth_candidates = [col for col in columns 
                                  if any(word in col.lower() for word in ['mode_researcher', 'answer', 'label', 'truth'])]
                
                default_truth = truth_candidates[0] if truth_candidates else columns[-1]
                
                ground_truth_column = st.selectbox(
                    "Select column containing ground truth labels:",
                    columns,
                    index=columns.index(default_truth) if default_truth in columns else len(columns)-1
                )
                
                st.session_state.ground_truth_column = ground_truth_column
                st.session_state.bot.ground_truth_column = ground_truth_column
            
            if st.button("Save Column Configuration"):
                st.success("✅ Column configuration saved!")
                
                # Show sample data with selected columns
                st.subheader("Sample Data Preview")
                sample_df = data[[text_column, ground_truth_column]].head()
                st.dataframe(sample_df)

    elif current_step == "4. Customize Prompt":
        st.header("✏️ Step 4: Customize Dictionary Generation Prompt")
        
        default_prompt = "Generate a list of single-word (unigram) keywords for a text classification dictionary focused on the 'tactic' based on the 'context'"
        
        custom_prompt = st.text_area(
            "Dictionary generation prompt:",
            value=st.session_state.get('custom_prompt', default_prompt),
            height=100,
            help="This prompt will be sent to Claude AI to generate keywords. Use 'tactic' and 'context' as placeholders."
        )
        
        if st.button("Save Prompt"):
            st.session_state.custom_prompt = custom_prompt
            st.session_state.bot.custom_prompt = custom_prompt
            st.success("✅ Prompt saved!")

    elif current_step == "5. Generate Dictionary":
        st.header("🤖 Step 5: Generate & Edit Dictionary")
        
        # Check prerequisites
        if not all(key in st.session_state for key in ['tactic', 'data', 'text_column']):
            st.warning("⚠️ Please complete previous steps first")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generate with Claude AI")
            
            if st.button("🤖 Generate Dictionary with AI"):
                with st.spinner("Generating dictionary with Claude AI..."):
                    # Prepare context
                    context = '. '.join(st.session_state.data[st.session_state.text_column].head(5).astype(str))
                    
                    keywords = st.session_state.bot.generate_ai_dictionary(
                        context, 
                        st.session_state.tactic,
                        st.session_state.get('custom_prompt', st.session_state.bot.custom_prompt)
                    )
                    
                    if keywords:
                        st.session_state.dictionary = keywords
                        st.session_state.bot.dictionary = keywords
                        st.success(f"✅ Generated {len(keywords)} keywords!")
                    else:
                        st.error("❌ Failed to generate dictionary with AI")
        
        with col2:
            st.subheader("Manual Entry")
            
            manual_keywords = st.text_area(
                "Enter keywords (comma-separated):",
                placeholder="classic,timeless,luxury,premium,sophisticated",
                height=100
            )
            
            if st.button("Add Manual Keywords"):
                if manual_keywords.strip():
                    keywords = [word.strip().strip('"\'') for word in manual_keywords.split(',')]
                    keywords = [word for word in keywords if word and word.replace('-', '').isalpha()]
                    
                    st.session_state.dictionary = keywords
                    st.session_state.bot.dictionary = keywords
                    st.success(f"✅ Added {len(keywords)} keywords!")
                else:
                    st.error("❌ Please enter some keywords")
        
        # Display and edit dictionary
        if 'dictionary' in st.session_state and st.session_state.dictionary:
            st.subheader(f"📚 Current Dictionary ({len(st.session_state.dictionary)} keywords)")
            
            # Display keywords in a grid
            cols = st.columns(4)
            for i, keyword in enumerate(st.session_state.dictionary):
                with cols[i % 4]:
                    st.write(f"• {keyword}")
            
            # Edit dictionary
            st.subheader("Edit Dictionary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_keyword = st.text_input("Add keyword:")
                if st.button("➕ Add"):
                    if new_keyword.strip() and new_keyword not in st.session_state.dictionary:
                        st.session_state.dictionary.append(new_keyword.strip())
                        st.session_state.bot.dictionary.append(new_keyword.strip())
                        st.rerun()
            
            with col2:
                remove_keyword = st.selectbox("Remove keyword:", st.session_state.dictionary)
                if st.button("➖ Remove"):
                    if remove_keyword in st.session_state.dictionary:
                        st.session_state.dictionary.remove(remove_keyword)
                        st.session_state.bot.dictionary.remove(remove_keyword)
                        st.rerun()

    elif current_step == "6. Run Classification":
        st.header("🚀 Step 6: Run Classification")
        
        # Check prerequisites
        required_keys = ['tactic', 'data', 'text_column', 'ground_truth_column', 'dictionary']
        if not all(key in st.session_state for key in required_keys):
            st.warning("⚠️ Please complete previous steps first")
            return
        
        if not st.session_state.dictionary:
            st.warning("⚠️ Dictionary is empty. Please add keywords in Step 5.")
            return
        
        if st.button("🚀 Run Classification"):
            with st.spinner("Running classification..."):
                # Run classification
                results, predictions = st.session_state.bot.classify_texts(
                    st.session_state.data,
                    st.session_state.text_column,
                    st.session_state.ground_truth_column,
                    st.session_state.dictionary
                )
                
                st.session_state.results = results
                st.session_state.bot.results = results
                
                # Calculate metrics
                metrics = st.session_state.bot.calculate_metrics(results)
                st.session_state.metrics = metrics
                
                # Display metrics
                st.subheader("📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.1f}%")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.1f}%")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.1f}%")
                
                st.success("✅ Classification completed!")

    elif current_step == "7. View Results":
        st.header("📈 Step 7: View Results & Analysis")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Please run classification first (Step 6)")
            return
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📝 Detailed Results", "🔍 Keyword Analysis", "💾 Download"])
        
        with tab1:
            st.subheader("Performance Summary")
            
            if 'metrics' in st.session_state:
                metrics = st.session_state.metrics
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🎯 Accuracy", f"{metrics['accuracy']:.1f}%")
                    st.metric("🔍 Precision", f"{metrics['precision']:.1f}%")
                
                with col2:
                    st.metric("📊 Recall", f"{metrics['recall']:.1f}%")
                    st.metric("⚖️ F1 Score", f"{metrics['f1']:.1f}%")
            
            # Show true positives
            true_positives = [r for r in st.session_state.results if r['prediction'] == 1 and r['ground_truth'] == 1]
            
            st.subheader(f"✅ True Positive Classifications ({len(true_positives)})")
            
            if true_positives:
                for i, result in enumerate(true_positives[:5], 1):
                    with st.expander(f"Example {i}: {result['text'][:60]}..."):
                        st.write(f"**Text:** {result['text']}")
                        st.write(f"**Matched Keywords:** {', '.join(result['matched_keywords'])}")
        
        with tab2:
            st.subheader("Detailed Classification Results")
            
            # Create results dataframe
            results_df = pd.DataFrame([
                {
                    'Text': r['text'],
                    'Ground Truth': r['ground_truth'],
                    'Prediction': r['prediction'],
                    'Matched Keywords': ', '.join(r['matched_keywords']),
                    'Keyword Count': r['keyword_count'],
                    'Correct': '✅' if r['prediction'] == r['ground_truth'] else '❌'
                }
                for r in st.session_state.results
            ])
            
            st.dataframe(results_df, use_container_width=True)
        
        with tab3:
            st.subheader("🔍 Keyword Performance Analysis")
            
            if st.button("🔄 Analyze Keywords"):
                with st.spinner("Analyzing keyword performance..."):
                    analysis = st.session_state.bot.analyze_keywords(
                        st.session_state.results,
                        st.session_state.dictionary
                    )
                    st.session_state.keyword_analysis = analysis
            
            if 'keyword_analysis' in st.session_state:
                analysis = st.session_state.keyword_analysis
                
                # Create keyword performance dataframe
                keyword_df = pd.DataFrame([
                    {
                        'Keyword': kw,
                        'Precision (%)': f"{data['precision']:.1f}",
                        'Recall (%)': f"{data['recall']:.1f}",
                        'F1 Score (%)': f"{data['f1']:.1f}",
                        'Total Matches': data['total_matches']
                    }
                    for kw, data in analysis.items()
                ]).sort_values('F1 Score (%)', ascending=False)
                
                st.dataframe(keyword_df, use_container_width=True)
        
        with tab4:
            st.subheader("💾 Download Results")
            
            if st.button("📥 Prepare Downloads"):
                # Detailed results
                detailed_df = pd.DataFrame([
                    {
                        'id': r['id'],
                        'text': r['text'],
                        'ground_truth': r['ground_truth'],
                        'prediction': r['prediction'],
                        'matched_keywords': ', '.join(r['matched_keywords']),
                        'keyword_count': r['keyword_count']
                    }
                    for r in st.session_state.results
                ])
                
                # Summary metrics
                if 'metrics' in st.session_state:
                    metrics = st.session_state.metrics
                    summary_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],
                        'Score': [
                            metrics['accuracy'] / 100,
                            metrics['precision'] / 100,
                            metrics['recall'] / 100,
                            metrics['f1'] / 100
                        ]
                    })
                
                # Convert to CSV
                detailed_csv = detailed_df.to_csv(index=False)
                summary_csv = summary_df.to_csv(index=False) if 'metrics' in st.session_state else ""
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download Detailed Results",
                        data=detailed_csv,
                        file_name=f"{st.session_state.get('tactic', 'classification')}_detailed_results.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if summary_csv:
                        st.download_button(
                            label="📊 Download Summary",
                            data=summary_csv,
                            file_name=f"{st.session_state.get('tactic', 'classification')}_summary.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()
