import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import io
import base64

# Mario Theme Colors
MARIO_COLORS = {
    'red': '#E60012',           # Mario's hat red
    'blue': '#0066CC',          # Mario's overalls blue
    'yellow': '#FFD700',        # Coins/stars yellow
    'green': '#00A652',         # Luigi green
    'brown': '#8B4513',         # Blocks brown
    'orange': '#FF8C00',        # Fire flower orange
    'purple': '#8A2BE2',        # Poison mushroom purple
    'light_blue': '#87CEEB',    # Sky blue
    'dark_red': '#8B0000',      # Dark red
    'white': '#FFFFFF',         # Cloud white
    'black': '#000000'          # Outline black
}

# Set page config
st.set_page_config(
    page_title="🍄 Mario's Text Analysis Castle",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Mario theme
st.markdown("""
<style>
    /* Mario-themed background and main styling */
    .main .block-container {
        background: linear-gradient(135deg, #87CEEB 0%, #B0E0E6 50%, #87CEEB 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        border: 4px solid #8B4513;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(45deg, #00A652 0%, #32CD32 25%, #00A652 50%, #228B22 75%, #00A652 100%);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFD700 0%, #FFA500 100%);
        border: 3px solid #8B4513;
        border-radius: 10px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #E60012;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        background: linear-gradient(45deg, #FFD700, #FFA500);
        padding: 20px;
        border-radius: 15px;
        border: 4px solid #8B4513;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        color: #0066CC;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 10px;
        border: 2px solid #0066CC;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Success and info boxes */
    .success-box {
        background: linear-gradient(45deg, #00A652, #32CD32);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #228B22;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .info-box {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #8B4513;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #8B4513;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #E60012, #FF4500);
        color: white;
        border: 3px solid #8B0000;
        border-radius: 10px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #FF4500, #E60012);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #8B4513;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #8B4513;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #8B4513;
    }
    
    /* Sidebar elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 2px solid #8B4513;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 2px solid #8B4513;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 5px;
        border: 2px solid #8B4513;
    }
    
    /* Add Mario coin animation */
    @keyframes coinFlip {
        0% { transform: rotateY(0deg); }
        100% { transform: rotateY(360deg); }
    }
    
    .coin {
        display: inline-block;
        animation: coinFlip 2s linear infinite;
    }
</style>
""", unsafe_allow_html=True)

def mario_style_plot():
    """Set Mario-themed plot style"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.weight': 'bold',
        'axes.facecolor': MARIO_COLORS['light_blue'],
        'figure.facecolor': MARIO_COLORS['white'],
        'axes.edgecolor': MARIO_COLORS['black'],
        'axes.linewidth': 3,
        'grid.color': MARIO_COLORS['white'],
        'grid.alpha': 0.7,
        'font.size': 10
    })

def count_dictionary_matches(text, dictionary):
    """Count matches for a specific dictionary in text"""
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    count = 0
    
    for term in dictionary:
        # Use word boundaries for exact matches
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        count += len(re.findall(pattern, text))
    
    return count

def analyze_text_data(df, text_column, dictionaries):
    """Analyze text data using dictionaries"""
    df_analyzed = df.copy()
    
    # Apply dictionary analysis
    for dict_name, dictionary in dictionaries.items():
        col_name = f'{dict_name}_count'
        df_analyzed[col_name] = df_analyzed[text_column].apply(
            lambda x: count_dictionary_matches(x, dictionary)
        )
    
    return df_analyzed

def create_mario_visualization(df, dictionaries):
    """Create Mario-themed visualizations"""
    mario_style_plot()
    
    # Count columns for plotting
    count_columns = [f'{dict_name}_count' for dict_name in dictionaries.keys()]
    
    if not count_columns:
        st.error("❌ No count columns found!")
        return None
    
    # Create figure with Mario-style layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('🍄 MARIO\'S SUPER TEXT ANALYTICS WORLD 🍄', 
                 fontsize=20, fontweight='bold', color=MARIO_COLORS['red'])
    
    # Add decorative border
    rect = Rectangle((0, 0), 1, 1, transform=fig.transFigure, 
                    linewidth=8, edgecolor=MARIO_COLORS['brown'], 
                    facecolor='none', zorder=1000)
    fig.patches.append(rect)
    
    # 1. Total Counts Bar Chart (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    totals = [df[col].sum() for col in count_columns]
    colors = [MARIO_COLORS['red'], MARIO_COLORS['green'], MARIO_COLORS['blue'], 
              MARIO_COLORS['orange'], MARIO_COLORS['purple']][:len(count_columns)]
    
    bars = ax1.bar(range(len(count_columns)), totals, color=colors, 
                   edgecolor=MARIO_COLORS['black'], linewidth=3)
    ax1.set_title('🎯 Total Matches by Category', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Dictionary Categories', fontweight='bold')
    ax1.set_ylabel('Total Matches', fontweight='bold')
    ax1.set_xticks(range(len(count_columns)))
    ax1.set_xticklabels([col.replace('_count', '').replace('_', ' ').title() 
                        for col in count_columns], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(total)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # 2. Distribution by Speaker (Top Middle) - if Speaker column exists
    if 'Speaker' in df.columns:
        ax2 = plt.subplot(2, 3, 2)
        speaker_counts = df.groupby('Speaker')[count_columns].sum()
        
        if len(speaker_counts) > 0:
            x_pos = np.arange(len(speaker_counts))
            width = 0.35 if len(count_columns) <= 2 else 0.2
            
            for i, col in enumerate(count_columns):
                color = colors[i % len(colors)]
                ax2.bar(x_pos + i*width, speaker_counts[col], width, 
                       label=col.replace('_count', '').replace('_', ' ').title(), 
                       color=color, edgecolor=MARIO_COLORS['black'], linewidth=2)
            
            ax2.set_title('👥 Matches by Speaker', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Speakers', fontweight='bold')
            ax2.set_ylabel('Match Count', fontweight='bold')
            ax2.set_xticks(x_pos + width * (len(count_columns)-1) / 2)
            ax2.set_xticklabels(speaker_counts.index)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    else:
        # Create a placeholder or alternative visualization
        ax2 = plt.subplot(2, 3, 2)
        ax2.text(0.5, 0.5, 'No Speaker Column\nFound in Data', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=16, fontweight='bold', color=MARIO_COLORS['red'])
        ax2.set_title('👥 Speaker Analysis', fontweight='bold', fontsize=12)
        ax2.axis('off')
    
    # 3. Match Distribution Histogram (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    total_matches_per_row = df[count_columns].sum(axis=1)
    ax3.hist(total_matches_per_row, bins=max(1, min(20, len(set(total_matches_per_row)))), 
             color=MARIO_COLORS['yellow'], edgecolor=MARIO_COLORS['black'], 
             linewidth=2, alpha=0.8)
    ax3.set_title('📊 Distribution of Total Matches', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Total Matches per Row', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    if len(df) > 1 and len(count_columns) > 1:
        # Create correlation matrix
        corr_matrix = df[count_columns].corr()
        im = ax4.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(count_columns)):
            for j in range(len(count_columns)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax4.set_title('🔥 Category Correlation Matrix', fontweight='bold', fontsize=12)
        ax4.set_xticks(range(len(count_columns)))
        ax4.set_yticks(range(len(count_columns)))
        ax4.set_xticklabels([col.replace('_count', '').replace('_', ' ').title() 
                            for col in count_columns], rotation=45)
        ax4.set_yticklabels([col.replace('_count', '').replace('_', ' ').title() 
                            for col in count_columns])
        plt.colorbar(im, ax=ax4, shrink=0.8)
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data\nfor Correlation', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=16, fontweight='bold', color=MARIO_COLORS['red'])
        ax4.set_title('🔥 Category Correlation Matrix', fontweight='bold', fontsize=12)
        ax4.axis('off')
    
    # 5. Time Series (if Turn column exists) (Bottom Middle)
    if 'Turn' in df.columns:
        ax5 = plt.subplot(2, 3, 5)
        for i, col in enumerate(count_columns):
            color = colors[i % len(colors)]
            ax5.plot(df['Turn'], df[col], marker='o', linewidth=3, 
                    markersize=6, label=col.replace('_count', '').replace('_', ' ').title(),
                    color=color, markeredgecolor=MARIO_COLORS['black'], 
                    markeredgewidth=1)
        
        ax5.set_title('📈 Matches Over Time (Turns)', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Turn Number', fontweight='bold')
        ax5.set_ylabel('Match Count', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5 = plt.subplot(2, 3, 5)
        ax5.text(0.5, 0.5, 'No Turn Column\nFound in Data', 
                ha='center', va='center', transform=ax5.transAxes,
                fontsize=16, fontweight='bold', color=MARIO_COLORS['red'])
        ax5.set_title('📈 Time Series Analysis', fontweight='bold', fontsize=12)
        ax5.axis('off')
    
    # 6. Summary Statistics (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = "🏆 POWER-UP STATISTICS 🏆\n\n"
    summary_text += f"📝 Total Rows Analyzed: {len(df)}\n"
    summary_text += f"📊 Dictionary Categories: {len(dictionaries)}\n\n"
    
    for col in count_columns:
        dict_name = col.replace('_count', '').replace('_', ' ').title()
        total = df[col].sum()
        avg = df[col].mean()
        max_val = df[col].max()
        summary_text += f"⭐ {dict_name}:\n"
        summary_text += f"   Total: {total} | Avg: {avg:.2f} | Max: {max_val}\n\n"
    
    # Find highest scoring row
    if len(count_columns) > 0:
        total_per_row = df[count_columns].sum(axis=1)
        if len(total_per_row) > 0:
            max_idx = total_per_row.idxmax()
            summary_text += f"🎯 Highest Scoring Row: #{max_idx}\n"
            summary_text += f"   Total Matches: {total_per_row[max_idx]}"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=MARIO_COLORS['yellow'], 
                      edgecolor=MARIO_COLORS['black'], linewidth=3, alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def main():
    # Header
    st.markdown('''
    <h1 class="main-header">
        🍄 MARIO'S TEXT ANALYSIS CASTLE 🍄
        <br><span class="coin">🪙</span> <span class="coin">⭐</span> <span class="coin">🪙</span>
    </h1>
    ''', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🎮 Welcome to the most super text analysis tool in the Mushroom Kingdom! 🎮</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("🎮 Configuration Panel")
    
    # Default dictionaries
    default_dictionaries = {
        'urgency_marketing': {
            'limited', 'limited time', 'limited run', 'limited edition', 'order now',
            'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
            'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
            'expires soon', 'final hours', 'almost gone'
        },
        'exclusive_marketing': {
            'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
            'members only', 'vip', 'special access', 'invitation only',
            'premium', 'privileged', 'limited access', 'select customers',
            'insider', 'private sale', 'early access'
        }
    }
    
    # File upload
    st.sidebar.subheader("📁 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your CSV file containing text data to analyze"
    )
    
    # Text column selection
    st.sidebar.subheader("📝 Select Text Column")
    st.sidebar.write("Choose the column containing text to analyze:")
    
    # Default column options when no file is uploaded
    default_columns = ["ID", "Turn", "Speaker", "Context", "Statement", "Tactic_human", "Tactic_human_reasoning"]
    
    # If file is uploaded, use actual columns, otherwise use defaults
    if uploaded_file is not None:
        try:
            # Read just the header to get column names
            temp_df = pd.read_csv(uploaded_file, nrows=0)
            available_columns = list(temp_df.columns)
            uploaded_file.seek(0)  # Reset file pointer
        except:
            available_columns = default_columns
    else:
        available_columns = default_columns
    
    # Find default selection
    if "Statement" in available_columns:
        default_index = available_columns.index("Statement")
    elif "Context" in available_columns:
        default_index = available_columns.index("Context")
    elif len(available_columns) > 0:
        default_index = 0
    else:
        default_index = 0
    
    text_column = st.sidebar.selectbox(
        "Select column:",
        options=available_columns,
        index=default_index,
        help="Choose the column that contains the text you want to analyze"
    )
    
    # Dictionary configuration
    st.sidebar.subheader("📚 Dictionary Configuration")
    
    # Use session state to manage dictionaries
    if 'dictionaries' not in st.session_state:
        st.session_state.dictionaries = default_dictionaries.copy()
    
    # Dictionary editor
    dict_option = st.sidebar.selectbox(
        "Choose action:",
        ["View/Edit Existing", "Add New Dictionary", "Reset to Default"]
    )
    
    if dict_option == "Reset to Default":
        if st.sidebar.button("🔄 Reset Dictionaries"):
            st.session_state.dictionaries = default_dictionaries.copy()
            st.sidebar.success("✅ Dictionaries reset to default!")
    
    elif dict_option == "Add New Dictionary":
        new_dict_name = st.sidebar.text_input("Dictionary Name:")
        new_dict_terms = st.sidebar.text_area(
            "Terms (one per line):",
            height=100,
            help="Enter each term on a new line"
        )
        
        if st.sidebar.button("➕ Add Dictionary"):
            if new_dict_name and new_dict_terms:
                terms = set(term.strip() for term in new_dict_terms.split('\n') if term.strip())
                st.session_state.dictionaries[new_dict_name] = terms
                st.sidebar.success(f"✅ Added '{new_dict_name}' dictionary with {len(terms)} terms!")
            else:
                st.sidebar.error("❌ Please provide both name and terms!")
    
    elif dict_option == "View/Edit Existing":
        if st.session_state.dictionaries:
            selected_dict = st.sidebar.selectbox(
                "Select dictionary to edit:",
                list(st.session_state.dictionaries.keys())
            )
            
            current_terms = '\n'.join(sorted(st.session_state.dictionaries[selected_dict]))
            edited_terms = st.sidebar.text_area(
                f"Edit {selected_dict} terms:",
                value=current_terms,
                height=150,
                help="Modify terms - one per line"
            )
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("💾 Save Changes"):
                    terms = set(term.strip() for term in edited_terms.split('\n') if term.strip())
                    st.session_state.dictionaries[selected_dict] = terms
                    st.sidebar.success("✅ Dictionary updated!")
            
            with col2:
                if st.button("🗑️ Delete Dict"):
                    del st.session_state.dictionaries[selected_dict]
                    st.sidebar.success("✅ Dictionary deleted!")
                    st.rerun()
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display data info
            st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", len(df))
            with col2:
                st.metric("📋 Total Columns", len(df.columns))
            with col3:
                st.metric("📚 Dictionaries", len(st.session_state.dictionaries))
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Check if text column exists
            if text_column not in df.columns:
                st.error(f"❌ Column '{text_column}' not found in the dataset!")
                st.info(f"Available columns: {', '.join(df.columns)}")
                return
            
            # Analysis button
            if st.button("🚀 Start Mario's Super Analysis!", type="primary"):
                if not st.session_state.dictionaries:
                    st.error("❌ No dictionaries configured! Please add at least one dictionary.")
                    return
                
                with st.spinner("🍄 Mario is analyzing your text... Please wait!"):
                    # Perform analysis
                    df_analyzed = analyze_text_data(df, text_column, st.session_state.dictionaries)
                    
                    # Display results summary
                    st.subheader("🎯 Analysis Results")
                    
                    # Create metrics for each dictionary
                    count_columns = [f'{dict_name}_count' for dict_name in st.session_state.dictionaries.keys()]
                    
                    if len(count_columns) <= 4:
                        cols = st.columns(len(count_columns))
                        for i, col_name in enumerate(count_columns):
                            dict_name = col_name.replace('_count', '').replace('_', ' ').title()
                            total_matches = df_analyzed[col_name].sum()
                            with cols[i]:
                                st.metric(f"⭐ {dict_name}", total_matches)
                    else:
                        # For more than 4 dictionaries, use a different layout
                        for col_name in count_columns:
                            dict_name = col_name.replace('_count', '').replace('_', ' ').title()
                            total_matches = df_analyzed[col_name].sum()
                            st.metric(f"⭐ {dict_name}", total_matches)
                    
                    # Create and display visualization
                    st.subheader("📊 Mario's Super Visualizations")
                    fig = create_mario_visualization(df_analyzed, st.session_state.dictionaries)
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)  # Close the figure to free memory
                    
                    # Show detailed results
                    st.subheader("📈 Detailed Results")
                    st.dataframe(df_analyzed)
                    
                    # Download button for results
                    csv_buffer = io.StringIO()
                    df_analyzed.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="💾 Download Analysis Results",
                        data=csv_data,
                        file_name="mario_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown('<div class="success-box">🎉 ANALYSIS COMPLETE! THANK YOU MARIO! 🎉</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ## 🎮 How to Use Mario's Text Analysis Castle:
        
        1. **📁 Upload Your Data**: Use the file uploader in the sidebar to upload a CSV file
        2. **📝 Set Text Column**: Specify which column contains the text you want to analyze
        3. **📚 Configure Dictionaries**: 
           - View and edit existing dictionaries
           - Add new dictionaries with custom terms
           - Reset to default marketing dictionaries
        4. **🚀 Run Analysis**: Click the analysis button to start Mario's super text analysis
        5. **📊 View Results**: Explore the Mario-themed visualizations and download your results
        
        ### 📋 Required CSV Format:
        Your CSV file should have at least one text column for analysis. Common column names include:
        - `Statement`, `Text`, `Content`, `Message`, `Description`
        
        Optional columns for enhanced analysis:
        - `Speaker` or `Author` (for speaker-based analysis)
        - `Turn` or `Time` (for time series analysis)
        
        ### 🍄 Default Dictionaries:
        """)
        
        for dict_name, terms in default_dictionaries.items():
            st.write(f"**{dict_name.replace('_', ' ').title()}**: {', '.join(sorted(list(terms)[:10]))}{'...' if len(terms) > 10 else ''}")

if __name__ == "__main__":
    main()
