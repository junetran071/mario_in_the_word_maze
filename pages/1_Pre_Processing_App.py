import streamlit as st
import pandas as pd
import nltk
import re
import emoji
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
import io
from typing import Dict, List, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Princess Peach's Instagram Preprocessor",
    page_icon="👑",
    layout="wide"
)

# Princess Peach theme CSS
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #FFE4E6 0%, #FDF2F8 25%, #FCEDF0 50%, #F9E7EC 75%, #FFE4E8 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #EC4899, #F472B6, #FB7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(236, 72, 153, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #FDF2F8 0%, #FCE7F3 100%);
        border-right: 2px solid #F9A8D4;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #EC4899, #F472B6);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.4);
    }
    
    /* Metric styling */
    .css-1xarl3l {
        background: linear-gradient(135deg, #FDF2F8, #FCE7F3);
        border-radius: 15px;
        border: 2px solid #F9A8D4;
        padding: 1rem;
    }
    
    /* File uploader styling */
    .css-1cpxqw2 {
        border: 2px dashed #EC4899;
        border-radius: 15px;
        background: linear-gradient(135deg, #FFFBFC, #FDF2F8);
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(90deg, #10B981, #34D399);
        border-radius: 10px;
    }
    
    /* Crown decoration */
    .crown-decoration {
        text-align: center;
        font-size: 2rem;
        margin: 1rem 0;
    }
    
    /* Princess welcome message */
    .princess-welcome {
        background: linear-gradient(135deg, #FDF2F8, #FCE7F3, #FFEEF0);
        border: 2px solid #F9A8D4;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(236, 72, 153, 0.2);
    }
    
    .princess-welcome h2 {
        color: #BE185D;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .princess-welcome p {
        color: #831843;
        font-size: 1.2rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Download NLTK data on first run
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        nltk.download('punkt', quiet=True)

class TextPreprocessor:
    """Configurable text preprocessing for Instagram captions"""
    
    def __init__(self, config: Dict = None):
        """Initialize with custom configuration"""
        # Default configuration
        self.default_config = {
            'remove_emojis': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_hashtags': True,
            'remove_mentions': True,
            'normalize_unicode': True,
            'add_period_if_missing': True,
            'padding_token': '[PAD]'
        }
        
        # Update with user config
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
    
    def clean_text(self, text: str) -> str:
        """Clean Instagram caption text based on configuration"""
        if not isinstance(text, str) or not text.strip():
            return self.config['padding_token']
        
        # Remove emojis
        if self.config['remove_emojis']:
            text = emoji.replace_emoji(text, replace='')
        
        # Normalize unicode characters
        if self.config['normalize_unicode']:
            text = unidecode(text)
        
        # Remove URLs
        if self.config['remove_urls']:
            text = self.url_pattern.sub('', text)
        
        # Remove emails
        if self.config['remove_emails']:
            text = self.email_pattern.sub('', text)
        
        # Remove hashtags
        if self.config['remove_hashtags']:
            text = self.hashtag_pattern.sub('', text)
        
        # Remove mentions
        if self.config['remove_mentions']:
            text = self.mention_pattern.sub('', text)
        
        # Clean whitespace
        text = text.replace('\n', ' ').strip()
        text = ' '.join(text.split())
        
        return text if text else self.config['padding_token']
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if text == self.config['padding_token'] or not text:
            return []
        
        # Add period if missing
        if self.config['add_period_if_missing'] and not text[-1] in '.!?':
            text += '.'
        
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

def process_dataframe(df: pd.DataFrame, config: Dict, required_columns: Dict) -> pd.DataFrame:
    """Process the uploaded dataframe"""
    preprocessor = TextPreprocessor(config)
    results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        # Update progress
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f'Processing row {idx + 1} of {len(df)}...')
        
        # Skip if no caption
        caption_col = required_columns['caption']
        if pd.isna(row[caption_col]):
            continue
        
        # Clean caption
        cleaned_caption = preprocessor.clean_text(row[caption_col])
        
        # Split into sentences
        sentences = preprocessor.split_sentences(cleaned_caption)
        
        # Create records for each sentence
        for turn, sentence in enumerate(sentences, 1):
            result_row = {
                'turn': turn,
                'caption': row[caption_col],  # Original caption (context)
                'transcript': sentence,       # Cleaned sentence (statement)
            }
            
            # Add other columns from original data
            for col_name, original_col in required_columns.items():
                if col_name != 'caption':
                    result_row[col_name] = row[original_col]
            
            results.append(result_row)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit app"""
    # Download NLTK data
    download_nltk_data()
    
    # Princess Peach welcome section
    st.markdown("""
    <div class="princess-welcome">
        <div class="crown-decoration">👑✨👑</div>
        <h2>Welcome to Princess Peach's Instagram</h2>
        <p>"Transform your royal captions into elegant sentence-level data with the power of the Mushroom Kingdom!"</p>
        <div class="crown-decoration">🌸💖🌸</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">👑 Royal Caption Preprocessor 👑</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.markdown("### 👑 Royal Preprocessing Settings")
    st.sidebar.markdown("*Customize your text transformation magic*")
    
    # Preprocessing options with princess-themed descriptions
    config = {}
    config['remove_emojis'] = st.sidebar.checkbox("✨ Remove Emojis", value=True, help="Remove emoji characters from text")
    config['remove_urls'] = st.sidebar.checkbox("🔗 Remove URLs", value=True, help="Remove web links")
    config['remove_emails'] = st.sidebar.checkbox("📧 Remove Emails", value=True, help="Remove email addresses")
    config['remove_hashtags'] = st.sidebar.checkbox("# Remove Hashtags", value=True, help="Remove hashtag symbols")
    config['remove_mentions'] = st.sidebar.checkbox("@ Remove Mentions", value=True, help="Remove @ mentions")
    config['normalize_unicode'] = st.sidebar.checkbox("🌟 Normalize Unicode", value=True, help="Convert special characters")
    config['add_period_if_missing'] = st.sidebar.checkbox("📝 Add Period if Missing", value=True, help="Add periods to complete sentences")
    config['padding_token'] = st.sidebar.text_input("💫 Padding Token", value="[PAD]", help="Token for empty content")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📜 Upload Your Royal Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file fit for a princess",
            type="csv",
            help="Upload your Instagram posts dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                st.success(f"✨ Successfully loaded {len(df)} royal posts and {len(df.columns)} data columns! ✨")
                
                # Show data preview
                st.markdown("## 👀 Royal Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column mapping
                st.markdown("## 🏰 Column Mapping")
                st.markdown("*Map your dataset columns to the royal requirements:*")
                
                available_columns = df.columns.tolist()
                
                required_columns = {}
                required_columns['shortcode'] = st.selectbox(
                    "👑 Royal Post ID Column", 
                    available_columns,
                    help="Unique identifier for each royal post"
                )
                required_columns['caption'] = st.selectbox(
                    "💬 Caption Column", 
                    available_columns,
                    help="Column containing the royal Instagram captions"
                )
                required_columns['post_url'] = st.selectbox(
                    "🔗 Post URL Column (Optional)", 
                    ['None'] + available_columns,
                    help="Column containing post URLs (optional)"
                )
                
                # Process button
                if st.button("🚀 Begin Royal Processing", type="primary"):
                    with st.spinner("Princess Peach is working her magic... ✨"):
                        # Filter out 'None' selections
                        filtered_columns = {k: v for k, v in required_columns.items() if v != 'None'}
                        
                        # Process the data
                        result_df = process_dataframe(df, config, filtered_columns)
                        
                        st.success(f"🎉 Royal success! Created {len(result_df)} elegant sentence records from {len(df)} posts! 👑")
                        
                        # Show results preview
                        st.markdown("## 📋 Your Royal Results")
                        st.dataframe(result_df.head(10), use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="💾 Download Your Royal Data",
                            data=csv_data,
                            file_name="princess_peach_ig_posts_preprocessed.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
                        # Show statistics
                        st.markdown("## 📊 Royal Processing Statistics")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("👑 Original Posts", len(df))
                        
                        with col_stat2:
                            st.metric("✨ Sentence Records", len(result_df))
                        
                        with col_stat3:
                            avg_sentences = len(result_df) / len(df) if len(df) > 0 else 0
                            st.metric("💫 Avg Sentences/Post", f"{avg_sentences:.2f}")
                        
            except Exception as e:
                st.error(f"👸 Oops! Princess Peach encountered an error: {str(e)}")
                st.info("💡 Please make sure your CSV file is properly formatted for the royal court!")
    
    with col2:
        st.markdown("## 📖 Royal Instructions")
        st.markdown("""
        ### 👑 How to use Princess Peach's Processor:
        
        1. **📜 Upload CSV**: Choose your royal Instagram posts dataset
        
        2. **⚙️ Configure**: Adjust magical preprocessing options in the sidebar
        
        3. **🗺️ Map Columns**: Select which columns contain your precious data
        
        4. **✨ Process**: Click the royal processing button to transform your data
        
        5. **💾 Download**: Save your elegantly processed results
        
        ### 📋 Expected Royal Input Format:
        Your CSV should contain at least:
        - **💬 Caption column**: Instagram post captions
        - **👑 ID column**: Unique post identifier
        - **🔗 URL column**: Post URLs (optional)
        
        ### 📊 Royal Output Format:
        - **turn**: Sentence number within each post
        - **caption**: Original royal caption text
        - **transcript**: Elegantly cleaned sentence
        - **Other columns**: Preserved from your input
        
        ### 🌸 Princess Peach's Tips:
        *"Remember, darling! Clean data is like a well-organized castle - everything has its perfect place!"* 👸✨
        """)
        
        st.markdown("## ⚙️ Current Royal Settings")
        st.json(config)

if __name__ == "__main__":
    main()
