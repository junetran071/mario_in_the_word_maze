"""
Instagram Caption Transformation Script - Fixed for Correct Output Format
Based on original by Dr. Yufan (Frank) Lin
Modified to output specific columns: ID, Sentence ID, Context, Statement
"""
!pip install emoji
!pip install unidecode
from unidecode import unidecode
import pandas as pd
import nltk
import re
import logging
import emoji
import csv
from typing import List, Dict
from google.colab import drive
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Class to handle text preprocessing operations for Instagram captions"""

    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#[\w\d]+')
        self.mention_pattern = re.compile(r'@[\w\d]+')

    def remove_emoji(self, text: str) -> str:
        """Remove emoji characters from text"""
        return emoji.replace_emoji(text, replace='')

    def clean_text(self, text: str) -> str:
        """Clean and normalize Instagram caption text"""
        if not isinstance(text, str):
            return "[PAD]"
        try:
            # Remove emojis first
            text = self.remove_emoji(text)

            # Normalize text with unidecode
            text = unidecode(text)

            # Remove URLs, email addresses, mentions and hashtags
            text = self.url_pattern.sub('', text)
            text = self.email_pattern.sub('', text)
            text = self.hashtag_pattern.sub('', text)
            text = self.mention_pattern.sub('', text)

            # Replace newlines with spaces
            text = text.replace('\n', ' ')

            # Remove extra spaces and strip
            text = ' '.join(text.split())

            return text if text.strip() else "[PAD]"
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return "[PAD]"

    def split_sentences(self, caption: str) -> List[str]:
        """
        Split caption text into sentences using NLTK

        Args:
            caption (str): Cleaned caption text

        Returns:
            List[str]: List of sentences
        """
        try:
            # Handle Instagram captions that might not end with proper punctuation
            if caption and not caption[-1] in '.!?':
                caption = caption + '.'

            sentences = sent_tokenize(caption)
            # Filter out empty sentences
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {str(e)}")
            return []

    def transform_caption_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Instagram caption dataframe into sentence-level data with correct output format"""
        transformed_rows = []

        for _, row in df.iterrows():
            # Skip if caption is missing
            if pd.isna(row['caption']):
                continue

            # Get the post ID (try different possible column names)
            post_id = ''
            if 'shortcode' in row and not pd.isna(row['shortcode']):
                post_id = str(row['shortcode'])
            elif 'post_id' in row and not pd.isna(row['post_id']):
                post_id = str(row['post_id'])
            elif 'id' in row and not pd.isna(row['id']):
                post_id = str(row['id'])
            else:
                # Use row index as fallback
                post_id = f"post_{len(transformed_rows)}"

            sentences = self.split_sentences(row['cleaned_caption'])
            for sentence_id, sentence in enumerate(sentences, 1):
                transformed_row = {
                    'ID': post_id,                    # Post identifier
                    'Sentence ID': sentence_id,       # Sentence number within post
                    'Context': row['caption'],        # Original caption as context
                    'Statement': sentence             # Cleaned sentence
                }
                transformed_rows.append(transformed_row)

        return pd.DataFrame(transformed_rows)

def verify_csv_reading(file_path: str) -> pd.DataFrame:
    """
    Verify that CSV rows are being read correctly

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame with the CSV data
    """
    try:
        # Try different encodings if the default doesn't work
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read CSV with {encoding} encoding, trying next...")

        if df is None:
            raise ValueError("Could not read CSV with any of the attempted encodings")

        # Verify that expected columns are present
        expected_columns = ['caption']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"CSV file must contain these columns: {expected_columns}")

        # Check for NaN values in critical columns
        critical_columns = ['caption']
        nan_counts = {col: df[col].isna().sum() for col in critical_columns if col in df.columns}

        if any(nan_counts.values()):
            logger.warning(f"NaN values found in critical columns: {nan_counts}")

        return df

    except Exception as e:
        logger.error(f"Error verifying CSV: {str(e)}")
        raise

def main():
    """Main function to execute the transformation pipeline"""
    try:
        # Mount Google Drive
        drive.mount('/content/drive')
        
        # Update these paths to match your file structure
        input_filename = "ig_posts_raw.csv"  # Change this to your actual input filename
        output_filename = "ig_posts_pre-processing.csv"  # This matches your required output name
        
        # You'll need to update this path to where your files are located
        base_path = "/content/drive/My Drive"  # Update this path as needed
        input_path = f"{base_path}/{input_filename}"
        output_path = f"{base_path}/{output_filename}"

        logger.info("Verifying and reading input data...")
        df = verify_csv_reading(input_path)

        # Print sample to verify correct reading
        logger.info(f"Read {len(df)} rows from CSV")
        logger.info(f"Columns found: {list(df.columns)}")
        if len(df) > 0:
            logger.info(f"Sample caption: {df['caption'].iloc[0][:100]}...")

        # Count captions with potential special characters
        emoji_pattern = re.compile(r'[^\w\s.,!?\'"()]')
        emojis_count = sum(df['caption'].apply(
            lambda x: bool(emoji_pattern.search(str(x))) if not pd.isna(x) else False
        ))
        logger.info(f"Number of captions with special characters: {emojis_count}")

        logger.info("Transforming caption data...")
        preprocessor = TextPreprocessor()
        df['cleaned_caption'] = df['caption'].apply(lambda x: preprocessor.clean_text(x))

        transformed_df = preprocessor.transform_caption_data(df)

        logger.info(f"Created {len(transformed_df)} sentence-level records from {len(df)} captions")

        # Verify output columns match expected format
        expected_output_columns = ['ID', 'Sentence ID', 'Context', 'Statement']
        actual_columns = list(transformed_df.columns)
        
        if actual_columns != expected_output_columns:
            logger.error(f"Output columns mismatch. Expected: {expected_output_columns}, Got: {actual_columns}")
        else:
            logger.info("Output format verified - columns match expected format")

        logger.info("Saving transformed data...")
        # Save with proper formatting
        transformed_df.to_csv(
            output_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,  # Quote all non-numeric fields
            quotechar='"',
            encoding='utf-8'
        )
        
        # Display sample of the output
        logger.info("Sample of transformed data:")
        print(transformed_df.head())
        
        logger.info(f"Transformation complete! Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
