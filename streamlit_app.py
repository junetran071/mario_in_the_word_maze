import streamlit as st
import pandas as pd
import re
import io
from typing import Dict, List, Optional

# Try to import optional dependencies with error handling
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Princess Peach's Instagram Pre-Processor",
    page_icon="👑",
    layout="wide"
)

# Princess Peach theme CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #EC4899, #F472B6, #FB7185);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .princess-welcome {
        background: linear-gradient(135deg, #FDF2F8, #FCE7F3, #FFEEF0);
        border: 2px solid #F9A8D4;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
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
    if not NLTK_AVAILABLE:
        return False
    
    try:
        import nltk
        nltk.download('punkt_tab', quiet=True)
        return True
    except:
        try:
            nltk.download('punkt', quiet=True)
            return True
        except:
            return False

class TextPre-Processor:
    """Configurable text pre-processing for Instagram captions"""
    
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
        if self.config['remove_emojis'] and EMOJI_AVAILABLE:
            text = emoji.replace_emoji(text, replace='')
        elif self.config['remove_emojis'] and not EMOJI_AVAILABLE:
            # Fallback: simple emoji removal using regex
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        
        # Normalize unicode characters
        if self.config['normalize_unicode'] and UNIDECODE_AVAILABLE:
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
        
        if NLTK_AVAILABLE:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        else:
            # Fallback: simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return [s.strip() for s in sentences if s.strip()]

def process_dataframe(df: pd.DataFrame, config: Dict, caption_column: str, id_column: str) -> pd.DataFrame:
    """Process the uploaded dataframe to match the required output format"""
    pre-processor = TextPreprocessor(config)
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
        if pd.isna(row[caption_column]):
            continue
        
        # Get the post ID
        post_id = ''
        if id_column and id_column in df.columns and not pd.isna(row[id_column]):
            post_id = str(row[id_column])
        else:
            post_id = f"post_{idx + 1}"
        
        # Clean caption
        cleaned_caption = preprocessor.clean_text(row[caption_column])
        
        # Split into sentences
        sentences = preprocessor.split_sentences(cleaned_caption)
        
        # Create records for each sentence in the required format
        for sentence_id, sentence in enumerate(sentences, 1):
            result_row = {
                'ID': post_id,                      # Post identifier
                'Sentence ID': sentence_id,         # Sentence number within post
                'Context': row[caption_column],     # Original caption as context
                'Statement': sentence               # Cleaned sentence as statement
            }
            results.append(result_row)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit app"""
    
    # Show dependency status
    deps_status = []
    if not NLTK_AVAILABLE:
        deps_status.append("⚠️ NLTK not available - using basic sentence splitting")
    if not EMOJI_AVAILABLE:
        deps_status.append("⚠️ Emoji package not available - using regex fallback")
    if not UNIDECODE_AVAILABLE:
        deps_status.append("⚠️ Unidecode not available - skipping unicode normalization")
    
    if deps_status:
        st.info("\n".join(deps_status))
        st.markdown("**To install missing packages:**")
        st.code("pip install nltk emoji unidecode")
    
    # Download NLTK data if available
    if NLTK_AVAILABLE:
        download_nltk_data()
    
    # Princess Peach welcome section
    st.markdown("""
    <div class="princess-welcome">
        <div style="font-size: 2rem; margin-bottom: 1rem;">👑✨👑</div>
        <h2>Welcome to Princess Peach's Instagram Pre-Processor</h2>
        <p>"Transform your royal captions into elegant sentence-level data!"</p>
        <div style="font-size: 2rem; margin-top: 1rem;">🌸💖🌸</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">👑 Royal Caption Pre-Processor 👑</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.markdown("### 👑 Royal Pre-Processing Settings")
    
    # Preprocessing options
    config = {}
    config['remove_emojis'] = st.sidebar.checkbox("✨ Remove Emojis", value=True)
    config['remove_urls'] = st.sidebar.checkbox("🔗 Remove URLs", value=True)
    config['remove_emails'] = st.sidebar.checkbox("📧 Remove Emails", value=True)
    config['remove_hashtags'] = st.sidebar.checkbox("# Remove Hashtags", value=True)
    config['remove_mentions'] = st.sidebar.checkbox("@ Remove Mentions", value=True)
    config['normalize_unicode'] = st.sidebar.checkbox("🌟 Normalize Unicode", value=True)
    config['add_period_if_missing'] = st.sidebar.checkbox("📝 Add Period if Missing", value=True)
    config['padding_token'] = st.sidebar.text_input("💫 Padding Token", value="[PAD]")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📜 Upload Your Royal Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your Instagram posts dataset in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                st.success(f"✨ Successfully loaded {len(df)} posts and {len(df.columns)} columns! ✨")
                
                # Show data preview
                st.markdown("## 👀 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column mapping
                st.markdown("## 🏰 Column Mapping")
                
                available_columns = df.columns.tolist()
                
                caption_column = st.selectbox(
                    "💬 Caption Column *", 
                    available_columns,
                    help="Column containing Instagram captions (required)"
                )
                
                id_column = st.selectbox(
                    "👑 ID Column (Optional)", 
                    ['None'] + available_columns,
                    help="Column containing post IDs (optional - will generate if not provided)"
                )
                
                if id_column == 'None':
                    id_column = None
                
                # Process button
                if st.button("🚀 Begin Royal Processing", type="primary"):
                    with st.spinner("Princess Peach is working her magic... ✨"):
                        # Process the data
                        result_df = process_dataframe(df, config, caption_column, id_column)
                        
                        st.success(f"🎉 Royal success! Created {len(result_df)} sentence records from {len(df)} posts! 👑")
                        
                        # Show results preview
                        st.markdown("## 📋 Your Royal Results")
                        st.dataframe(result_df.head(10), use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="💾 Download ig_posts_pre-processing.csv",
                            data=csv_data,
                            file_name="ig_posts_pre-processing.csv",
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
                st.error(f"👸 Error: {str(e)}")
                st.info("💡 Please make sure your CSV file is properly formatted!")
    
    with col2:
        st.markdown("## 📖 Royal Instructions")
        st.markdown("""
        ### 👑 How to use:
        
        1. **📜 Upload CSV**: Choose your Instagram posts dataset
        
        2. **⚙️ Configure**: Adjust pre-processing options in the sidebar
        
        3. **🗺️ Map Columns**: Select caption and ID columns
        
        4. **✨ Process**: Click to transform your data
        
        5. **💾 Download**: Save your processed results
        
        ### 📊 Output Format:
        - **ID**: Post identifier
        - **Sentence ID**: Sentence number within post
        - **Context**: Original caption text
        - **Statement**: Cleaned individual sentence
        
        This matches the format in your sample file!
        """)
        
        st.markdown("## ⚙️ Current Settings")
        for key, value in config.items():
            st.text(f"{key}: {value}")

if __name__ == "__main__":
    main()
