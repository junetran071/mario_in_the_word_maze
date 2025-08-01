import streamlit as st
import pandas as pd
import io
from functools import reduce

# Page configuration
st.set_page_config(
    page_title="Doraemon's CSV Join Tool",
    page_icon="🐱",
    layout="wide"
)

# Doraemon theme CSS
st.markdown("""
<style>
    /* Main background with Doraemon light blue */
    .stApp {
        background: linear-gradient(135deg, #B3E5FC 0%, #E3F2FD 50%, #B3E5FC 100%);
    }
    
    /* Header styling with Doraemon colors */
    .main-header {
        background: linear-gradient(90deg, #1976D2, #2196F3, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(25, 118, 210, 0.3);
    }
    
    /* Instructions box styling */
    .instructions-box {
        background: white;
        border: 2px solid #1976D2;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Error box styling */
    .error-box {
        background-color: #FFEBEE;
        border: 2px solid #EF5350;
        border-radius: 10px;
        padding: 1rem;
        color: #C62828;
        margin: 1rem 0;
    }
    
    /* Success box styling */
    .success-box {
        background-color: #E8F5E9;
        border: 2px solid #66BB6A;
        border-radius: 10px;
        padding: 1rem;
        color: #2E7D32;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #1976D2, #2196F3);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: white;
        border: 2px dashed #1976D2;
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1976D2;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #E3F2FD 0%, #BBDEFB 100%);
    }
    
    /* Step numbering - Fixed size and alignment */
    .step-number {
        background: #1976D2;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
        font-size: 16px;
    }
    
    /* Step header container */
    .step-header {
        display: flex;
        align-items: center;
        margin: 1.5rem 0 1rem 0;
    }
    
    .step-header h3 {
        margin: 0;
        font-size: 1.5rem;
    }
    
    /* Tips box */
    .tips-box {
        background: #E1F5FE;
        border-left: 4px solid #03A9F4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Features list */
    .features-list {
        margin: 1rem 0;
    }
    
    .features-list ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .features-list li {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .features-list li:before {
        content: '•';
        color: #1976D2;
        font-size: 1.5em;
        position: absolute;
        left: 0;
    }
</style>
""", unsafe_allow_html=True)

def validate_file(uploaded_file):
    """
    Validate uploaded CSV file size and format
    Returns (is_valid, error_message, dataframe)
    """
    if uploaded_file is None:
        return False, "No file uploaded", None
    
    # Check file size (10 MB limit)
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB in bytes
    if uploaded_file.size > MAX_SIZE:
        return False, f"File size exceeds 10 MB limit: {uploaded_file.size / 1024 / 1024:.2f} MB", None
    
    try:
        df = pd.read_csv(uploaded_file)
        return True, "", df
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}", None

def multi_join(base_df: pd.DataFrame, dataframes: list, join_keys: list, join_types: list) -> pd.DataFrame:
    """
    Join multiple dataframes to a base dataframe simultaneously
    
    Args:
        base_df: The base dataframe to join others to
        dataframes: List of dataframes to join
        join_keys: List of tuples containing (base_keys, other_keys) for each join
        join_types: List of join types for each operation
    
    Returns:
        Joined dataframe
    """
    result = base_df.copy()
    
    for df, keys, join_type in zip(dataframes, join_keys, join_types):
        base_keys, other_keys = keys
        # Create dictionary mapping keys
        key_pairs = dict(zip(base_keys, other_keys))
        # Rename columns in other dataframe to match base
        df_renamed = df.rename(columns={v: k for k, v in key_pairs.items()})
        # Perform join
        result = pd.merge(
            result,
            df_renamed,
            on=base_keys,
            how=join_type
        )
    
    return result

def main():
    # Welcome header with Doraemon theme
    st.markdown("<h1 class='main-header'>🐱 Doraemon's Magical CSV Join Tool 🔧</h1>", unsafe_allow_html=True)
    
    # Instructions section with proper markdown
    st.markdown("### Welcome to Doraemon's CSV Join Tool!")
    
    # What This Tool Can Do section
    st.markdown("#### 🎯 What This Tool Can Do:")
    st.markdown("""
    - Join two CSV files using different methods (inner, left, right, outer)
    - Handle files up to 10 MB each
    - Preview and validate your data before joining
    - Download the combined results
    """)
    
    # Tips section
    st.markdown("#### 💡 Doraemon's Tips:")
    st.markdown("""
    - Make sure your CSV files have at least one common column for joining
    - Check the first few rows of your data before joining
    - Choose the right join type based on your needs
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # File upload section
    st.markdown("<div class='step-header'>", unsafe_allow_html=True)
    st.markdown("<span class='step-number'>1</span> Upload Your CSV Files", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("Let's start by uploading your CSV files! Each file should be 10 MB or smaller.")
    
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload 2 or more CSV files (max 10 MB each)"
    )
    
    if len(uploaded_files) < 2:
        st.warning("🤖 Please upload at least 2 CSV files to use the magical join tool!")
        return
    
    # Validate files and store valid ones
    valid_files = {}
    for uploaded_file in uploaded_files:
        is_valid, error_msg, df = validate_file(uploaded_file)
        if is_valid:
            valid_files[uploaded_file.name] = df
        else:
            st.markdown(f"<div class='error-box'>❌ {uploaded_file.name}: {error_msg}</div>", unsafe_allow_html=True)
    
    if len(valid_files) < 2:
        st.error("❌ I need at least 2 valid CSV files to perform the join!")
        return
    
    # Display list of uploaded files
    st.markdown("""
        <div class='step-header'>
            <span class='step-number'>2</span>
            <h3>Check Your Files</h3>
        </div>
        <p>Great! Let's look at what you've uploaded. Here's a preview of each file:</p>
    """, unsafe_allow_html=True)
    
    for filename, df in valid_files.items():
        st.markdown(f"""
            <div style='background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 2px solid #1976D2;'>
                <h4>📄 {filename}</h4>
                <p><strong>Available Columns:</strong> {', '.join(df.columns.tolist())}</p>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(3), use_container_width=True)
    
    # Join configuration
    st.markdown("""
        <div class='step-header'>
            <span class='step-number'>3</span>
            <h3>Configure Multi-Join</h3>
        </div>
        <p>Now, let's set up how you want to join these files. Choose a base file and configure how other files should join to it!</p>
    """, unsafe_allow_html=True)
    
    # Select base file
    base_file = st.selectbox(
        "Select base file (other files will join to this one)",
        options=list(valid_files.keys()),
        help="This is the main file that others will join to"
    )
    
    # Select files to join
    other_files = st.multiselect(
        "Select files to join with base file",
        options=[f for f in valid_files.keys() if f != base_file],
        help="You can select multiple files to join simultaneously"
    )
    
    if not other_files:
        st.warning("🤖 Please select at least one file to join with the base file!")
        return
    
    # Configure joins
    join_configs = []
    
    for idx, other_file in enumerate(other_files):
        st.markdown(f"### Join Configuration for {other_file}")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Keys from base file
            base_keys = st.multiselect(
                f"Select join keys from {base_file}",
                options=valid_files[base_file].columns.tolist(),
                key=f"base_keys_{idx}"
            )
        
        with col2:
            # Keys from other file
            other_keys = st.multiselect(
                f"Select join keys from {other_file}",
                options=valid_files[other_file].columns.tolist(),
                key=f"other_keys_{idx}"
            )
        
        with col3:
            # Join type
            join_type = st.selectbox(
                "Join type",
                options=["inner", "left", "right", "outer"],
                key=f"join_type_{idx}"
            )
        
        if len(base_keys) != len(other_keys):
            st.error(f"❌ Number of keys must match for {other_file}!")
            return
        
        if not base_keys or not other_keys:
            st.warning(f"⚠️ Please select join keys for {other_file}")
            return
        
        join_configs.append({
            'file': other_file,
            'base_keys': base_keys,
            'other_keys': other_keys,
            'join_type': join_type
        })
    
    # Perform multi-join when button is clicked
    if st.button("🔮 Perform Magical Multi-Join!", type="primary"):
        try:
            # Prepare join parameters
            base_df = valid_files[base_file]
            other_dfs = [valid_files[cfg['file']] for cfg in join_configs]
            join_keys = [(cfg['base_keys'], cfg['other_keys']) for cfg in join_configs]
            join_types = [cfg['join_type'] for cfg in join_configs]
            
            # Perform multi-join
            result_df = multi_join(base_df, other_dfs, join_keys, join_types)
            
            # Show results
            st.markdown("""
                <div class='step-header'>
                    <span class='step-number'>4</span>
                    <h3>Join Results</h3>
                </div>
                <p>Ta-da! Here's what I created with my magical join tool:</p>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div class='success-box'>✨ Successfully joined all files! Created {len(result_df)} rows.</div>", unsafe_allow_html=True)
            st.dataframe(result_df.head(10), use_container_width=True)
            
            # Download option
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Your Magical Results!",
                data=csv_data,
                file_name="doraemon_joined_data.csv",
                mime="text/csv"
            )
            
            # Display statistics
            st.markdown("""
                <div class='step-header'>
                    <span class='step-number'>5</span>
                    <h3>Join Statistics</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Create columns for all files plus result
            stats_cols = st.columns(len(valid_files) + 1)
            
            # Show original file sizes
            for i, (filename, df) in enumerate(valid_files.items()):
                with stats_cols[i]:
                    st.metric(
                        f"{'📌 ' if filename == base_file else ''}Rows in {filename}", 
                        len(df)
                    )
            
            # Show final result size
            with stats_cols[-1]:
                st.metric("Final Result Rows", len(result_df))
            
        except Exception as e:
            st.markdown(f"<div class='error-box'>❌ Error performing joins: {str(e)}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

