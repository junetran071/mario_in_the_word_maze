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

class DictionaryClassifier:
    """
    Minimalist Dictionary-based Text Classification Tool for Google Colab
    """
    
    def __init__(self):
        self.data = None
        self.dictionary = []
        self.text_column = None
        self.ground_truth_column = None
        self.results = None
        self.keyword_analysis = None
        
    def load_data(self, file_path: str = None, csv_text: str = None):
        """Load CSV data from file or text string"""
        if file_path:
            self.data = pd.read_csv(file_path)
        elif csv_text:
            from io import StringIO
            self.data = pd.read_csv(StringIO(csv_text))
        else:
            # Load sample data
            sample_data = """ID,Statement,Answer
1,Its SPRING TRUNK SHOW week!,1
2,I am offering 4 shirts styled the way you want and the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!,1
3,In recognition of Earth Day I would like to showcase our collection of Earth Fibers!,0
4,It is now time to do some wardrobe crunches and check your basics! Never on sale.,1
5,He is a hard worker and always willing to lend a hand. The prices are the best I have seen in 17 years of servicing my clients.,0"""
            from io import StringIO
            self.data = pd.read_csv(StringIO(sample_data))
            
        # Auto-detect columns
        columns = self.data.columns.tolist()
        if 'Statement' in columns:
            self.text_column = 'Statement'
        elif 'statement' in columns:
            self.text_column = 'statement'
        else:
            # Use first text-heavy column
            for col in columns:
                if self.data[col].dtype == 'object':
                    avg_length = self.data[col].astype(str).str.len().mean()
                    if avg_length > 20:  # Likely text column
                        self.text_column = col
                        break
                        
        if 'Answer' in columns:
            self.ground_truth_column = 'Answer'
        elif 'answer' in columns:
            self.ground_truth_column = 'answer'
        elif 'label' in columns:
            self.ground_truth_column = 'label'
            
        print(f"📊 Loaded {len(self.data)} rows")
        print(f"📝 Text column: {self.text_column}")
        print(f"🎯 Ground truth column: {self.ground_truth_column}")
        print("\n🔍 Data preview:")
        print(self.data.head())
        
    def set_dictionary(self, keywords: List[str]):
        """Set the keyword dictionary"""
        self.dictionary = [kw.strip().lower() for kw in keywords if kw.strip()]
        print(f"📚 Dictionary set with {len(self.dictionary)} keywords:")
        print(f"✨ Keywords: {', '.join(self.dictionary)}")
        
    def classify(self):
        """Perform classification using the dictionary"""
        if not self.dictionary:
            raise ValueError("❌ Please set dictionary first using set_dictionary()")
        if self.text_column not in self.data.columns:
            raise ValueError(f"❌ Text column '{self.text_column}' not found")
            
        results = []
        
        for idx, row in self.data.iterrows():
            text = str(row[self.text_column]).lower()
            
            # Find matched keywords
            matched_keywords = []
            keyword_frequencies = []
            
            for keyword in self.dictionary:
                if keyword in text:
                    matched_keywords.append(keyword)
                    # Count frequency of this keyword
                    freq = len(re.findall(re.escape(keyword), text))
                    keyword_frequencies.append(freq)
                else:
                    keyword_frequencies.append(0)
            
            # Binary prediction
            binary_prediction = 1 if matched_keywords else 0
            
            # Continuous scores
            continuous_score = len(matched_keywords) / len(self.dictionary)
            frequency_score = sum(keyword_frequencies)
            
            # Ground truth if available
            ground_truth = None
            if self.ground_truth_column and self.ground_truth_column in self.data.columns:
                try:
                    ground_truth = int(row[self.ground_truth_column])
                except:
                    ground_truth = None
            
            results.append({
                'text': row[self.text_column],
                'binary_prediction': binary_prediction,
                'continuous_score': continuous_score,
                'frequency_score': frequency_score,
                'matched_keywords': matched_keywords,
                'keyword_frequencies': keyword_frequencies,
                'ground_truth': ground_truth
            })
        
        self.results = pd.DataFrame(results)
        print(f"👸 Classification complete! Processed {len(self.results)} statements")
        
        # Calculate metrics if ground truth available
        if self.ground_truth_column:
            self._calculate_metrics()
            
        return self.results
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if self.results is None or self.ground_truth_column is None:
            return
            
        # Filter out rows with valid ground truth
        valid_results = self.results[self.results['ground_truth'].notna()].copy()
        
        if len(valid_results) == 0:
            print("⚠️ No valid ground truth data found")
            return
            
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
        
        print("\n" + "="*60)
        print("👑 ROYAL CLASSIFICATION RESULTS")
        print("="*60)
        print(f"🎯 Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"💎 Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"🌟 Recall:    {recall:.3f} ({recall*100:.1f}%)")
        print(f"👸 F1 Score:  {f1:.3f} ({f1*100:.1f}%)")
        print()
        print("🔮 Advanced Metrics:")
        print(f"📏 MAE:  {mae:.4f}")
        print(f"📐 RMSE: {rmse:.4f}")
        print(f"💫 R²:   {r2:.4f}")
        print(f"🌊 Correlation: {correlation:.4f}")
        print()
        print("📊 Confusion Matrix:")
        print(f"✅ True Positives:  {tp}")
        print(f"❌ False Positives: {fp}")
        print(f"💔 False Negatives: {fn}")
        print(f"✅ True Negatives:  {tn}")
        
        # Store metrics for later use
        self.metrics = {
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1,
            'mae': mae, 'rmse': rmse, 'r2_score': r2, 'correlation': correlation,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    def analyze_keywords(self):
        """Analyze individual keyword performance"""
        if self.results is None or self.ground_truth_column is None:
            print("❌ Need classification results and ground truth for keyword analysis")
            return
            
        valid_results = self.results[self.results['ground_truth'].notna()].copy()
        keyword_metrics = []
        
        for i, keyword in enumerate(self.dictionary):
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
        
        self.keyword_analysis = pd.DataFrame(keyword_metrics)
        
        # Display results
        print("\n" + "="*60)
        print("🔍 ROYAL KEYWORD ANALYSIS")
        print("="*60)
        
        # Top by recall
        print("\n🌟 TOP KEYWORDS BY RECALL (Catch Rate):")
        top_recall = self.keyword_analysis.nlargest(5, 'recall')
        for _, row in top_recall.iterrows():
            print(f"✨ '{row['keyword']}' - Recall: {row['recall']:.1%}, Precision: {row['precision']:.1%}, F1: {row['f1_score']:.1%}")
        
        # Top by precision  
        print("\n💎 TOP KEYWORDS BY PRECISION (Accuracy):")
        top_precision = self.keyword_analysis.nlargest(5, 'precision')
        for _, row in top_precision.iterrows():
            print(f"✨ '{row['keyword']}' - Precision: {row['precision']:.1%}, Recall: {row['recall']:.1%}, F1: {row['f1_score']:.1%}")
            
        return self.keyword_analysis
    
    def show_errors(self, error_type='both', max_examples=10):
        """Display false positives and/or false negatives"""
        if self.results is None:
            print("❌ No classification results available")
            return
            
        valid_results = self.results[self.results['ground_truth'].notna()].copy()
        
        if error_type in ['both', 'false_positives']:
            fp = valid_results[(valid_results['binary_prediction'] == 1) & (valid_results['ground_truth'] == 0)]
            print(f"\n🚫 FALSE POSITIVES ({len(fp)} total):")
            print("-" * 50)
            for i, (_, row) in enumerate(fp.head(max_examples).iterrows()):
                print(f"{i+1}. {row['text'][:100]}...")
                print(f"   💎 Matched: {', '.join(row['matched_keywords'])}")
                print(f"   📊 Score: {row['continuous_score']:.3f}\n")
        
        if error_type in ['both', 'false_negatives']:
            fn = valid_results[(valid_results['binary_prediction'] == 0) & (valid_results['ground_truth'] == 1)]
            print(f"\n💔 FALSE NEGATIVES ({len(fn)} total):")
            print("-" * 50)
            for i, (_, row) in enumerate(fn.head(max_examples).iterrows()):
                print(f"{i+1}. {row['text'][:100]}...")
                print(f"   🔍 No keywords matched")
                print(f"   📊 Score: {row['continuous_score']:.3f}\n")
    
    def export_results(self, filename='classification_results.csv'):
        """Export classification results to CSV"""
        if self.results is None:
            print("❌ No results to export")
            return
            
        export_df = self.results.copy()
        export_df['matched_keywords_str'] = export_df['matched_keywords'].apply(lambda x: ', '.join(x))
        export_df = export_df.drop('matched_keywords', axis=1)
        
        export_df.to_csv(filename, index=False)
        print(f"📥 Results exported to {filename}")
    
    def plot_metrics(self):
        """Create visualizations of the results"""
        if self.results is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('👑 Royal Classification Analysis', fontsize=16, fontweight='bold')
        
        # 1. Score distribution
        axes[0, 0].hist(self.results['continuous_score'], bins=20, alpha=0.7, color='pink', edgecolor='purple')
        axes[0, 0].set_title('🌸 Continuous Score Distribution')
        axes[0, 0].set_xlabel('Continuous Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Binary predictions
        binary_counts = self.results['binary_prediction'].value_counts()
        axes[0, 1].pie(binary_counts.values, labels=['Negative (0)', 'Positive (1)'], 
                       colors=['lightblue', 'lightpink'], autopct='%1.1f%%')
        axes[0, 1].set_title('👸 Binary Classification Results')
        
        # 3. Keyword frequency distribution
        if self.keyword_analysis is not None:
            top_keywords = self.keyword_analysis.nlargest(10, 'f1_score')
            axes[1, 0].barh(top_keywords['keyword'], top_keywords['f1_score'], color='mediumpurple')
            axes[1, 0].set_title('✨ Top Keywords by F1 Score')
            axes[1, 0].set_xlabel('F1 Score')
        
        # 4. Confusion matrix (if ground truth available)
        if hasattr(self, 'metrics'):
            cm_data = [[self.metrics['tn'], self.metrics['fp']], 
                      [self.metrics['fn'], self.metrics['tp']]]
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='RdYlBu_r', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'], ax=axes[1, 1])
            axes[1, 1].set_title('🎯 Confusion Matrix')
        
        plt.tight_layout()
        plt.show()

print("🌸 Princess Dictionary Classification Tool Ready! 🌸")


# Quick test
classifier = DictionaryClassifier()
classifier.load_data()  # Load sample data
classifier.set_dictionary(['spring', 'trunk', 'show', 'sale'])
classifier.classify()
classifier.analyze_keywords()
classifier.show_errors()
classifier.plot_metrics()
