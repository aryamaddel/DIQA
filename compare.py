#!/usr/bin/env python3
"""
Comprehensive Image Quality Assessment Comparison Tool
Compares BRISQUE, PIQE, NIQE, and other available metrics on KONIQ-10k dataset.

Note: CONTRIQUE and TRIQA will be added when available in pyiqa or implemented separately.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
from PIL import Image

try:
    import pyiqa
    import torch
    print(f"PyIQA version: {pyiqa.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Error importing required packages: {e}")
    sys.exit(1)


class ImageQualityAssessor:
    """Image Quality Assessment using multiple metrics."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize the assessor with specified device."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize available metrics
        self.metrics = {}
        self.metric_names = []
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all available IQA metrics."""
        # Core metrics that should be available
        core_metrics = {
            'brisque': 'brisque',
            'piqe': 'piqe', 
            'niqe': 'niqe',
            'ilniqe': 'ilniqe'
        }
        
        # Additional modern metrics to try
        additional_metrics = {
            'clipiqa': 'clipiqa',
            'clipiqa+': 'clipiqa+',
            'maniqa': 'maniqa',
            'musiq': 'musiq',
            'hyperiqa': 'hyperiqa',
            'cnniqa': 'cnniqa',
            'dbcnn': 'dbcnn'
        }
        
        # Try to initialize core metrics
        for name, model_name in core_metrics.items():
            try:
                metric = pyiqa.create_metric(model_name, device=self.device)
                self.metrics[name] = metric
                self.metric_names.append(name)
                print(f"✓ Initialized {name.upper()}")
            except Exception as e:
                print(f"✗ Failed to initialize {name.upper()}: {e}")
        
        # Try to initialize additional metrics
        for name, model_name in additional_metrics.items():
            try:
                metric = pyiqa.create_metric(model_name, device=self.device)
                self.metrics[name] = metric
                self.metric_names.append(name)
                print(f"✓ Initialized {name.upper()}")
            except Exception as e:
                print(f"✗ Failed to initialize {name.upper()}: {e}")
        
        if not self.metrics:
            raise RuntimeError("No metrics could be initialized!")
            
        print(f"\nSuccessfully initialized {len(self.metrics)} metrics: {', '.join(self.metric_names)}")
    
    def assess_image(self, image_path: str) -> Dict[str, float]:
        """Assess a single image with all available metrics."""
        results = {'image_path': image_path}
        
        try:
            # Verify image exists and is readable
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Try to open with PIL first to verify it's a valid image
            with Image.open(image_path) as img:
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
            
            # Assess with each metric
            for metric_name, metric in self.metrics.items():
                try:
                    score = metric(image_path)
                    if torch.is_tensor(score):
                        score = score.item()
                    results[metric_name] = float(score)
                except Exception as e:
                    print(f"Error computing {metric_name} for {image_path}: {e}")
                    results[metric_name] = np.nan
                    
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            for metric_name in self.metric_names:
                results[metric_name] = np.nan
        
        return results
    
    def assess_directory(self, image_dir: str, output_file: str = None, 
                        sample_size: int = None, save_interval: int = 100) -> pd.DataFrame:
        """Assess all images in a directory."""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Find all image files (avoid duplicates by using case-insensitive matching)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            # Only search for lowercase extensions to avoid duplicates
            image_files.extend(list(image_dir.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        if sample_size and sample_size < len(image_files):
            image_files = np.random.choice(image_files, sample_size, replace=False)
            print(f"Randomly selected {len(image_files)} images for processing")
        
        # Initialize results list
        results = []
        
        # Setup output file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process images with progress bar
        start_time = time.time()
        for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
            result = self.assess_image(str(image_file))
            result['filename'] = image_file.name
            results.append(result)
            
            # Save intermediate results
            if output_file and (i + 1) % save_interval == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(output_file, index=False)
                print(f"Saved intermediate results after {i + 1} images")
        
        # Create final dataframe
        df = pd.DataFrame(results)
        
        # Save final results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Final results saved to {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/len(image_files):.3f} seconds")
        
        return df


class ResultAnalyzer:
    """Analyze and visualize IQA results."""
    
    def __init__(self, results_df: pd.DataFrame):
        """Initialize with results dataframe."""
        self.df = results_df.copy()
        self.metric_columns = [col for col in self.df.columns 
                              if col not in ['image_path', 'filename']]
        
    def print_statistics(self):
        """Print basic statistics for all metrics."""
        print("\n" + "="*60)
        print("IMAGE QUALITY ASSESSMENT STATISTICS")
        print("="*60)
        
        print(f"\nDataset size: {len(self.df)} images")
        print(f"Metrics evaluated: {len(self.metric_columns)}")
        print(f"Metrics: {', '.join(self.metric_columns)}")
        
        print("\nDescriptive Statistics:")
        print("-" * 40)
        stats = self.df[self.metric_columns].describe()
        print(stats.round(4))
        
        print("\nMissing Values:")
        print("-" * 20)
        missing = self.df[self.metric_columns].isnull().sum()
        for metric, count in missing.items():
            percentage = (count / len(self.df)) * 100
            print(f"{metric}: {count} ({percentage:.1f}%)")
    
    def correlation_analysis(self):
        """Analyze correlations between metrics."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Remove rows with any NaN values for correlation analysis
        clean_df = self.df[self.metric_columns].dropna()
        print(f"\nUsing {len(clean_df)} images for correlation analysis")
        
        if len(clean_df) < 2:
            print("Not enough clean data for correlation analysis")
            return
        
        corr_matrix = clean_df.corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
        # Find highest correlations
        print("\nHighest correlations (excluding self-correlations):")
        print("-" * 50)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                metric1 = corr_matrix.columns[i]
                metric2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                print(f"{metric1} vs {metric2}: {corr_val:.4f}")
    
    def create_visualizations(self, output_dir: str = "results"):
        """Create comprehensive visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribution plots
        self._plot_distributions(output_path)
        
        # 2. Correlation heatmap
        self._plot_correlation_heatmap(output_path)
        
        # 3. Scatter plots
        self._plot_scatter_matrix(output_path)
        
        # 4. Box plots
        self._plot_boxplots(output_path)
        
        print(f"\nVisualization plots saved in {output_path}")
    
    def _plot_distributions(self, output_path: Path):
        """Plot distribution of each metric."""
        n_metrics = len(self.metric_columns)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(self.metric_columns):
            clean_data = self.df[metric].dropna()
            if len(clean_data) > 0:
                axes[i].hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.upper()} Distribution')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.metric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, output_path: Path):
        """Plot correlation heatmap."""
        clean_df = self.df[self.metric_columns].dropna()
        if len(clean_df) < 2:
            return
        
        corr_matrix = clean_df.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Metric Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scatter_matrix(self, output_path: Path):
        """Plot scatter matrix for all metric pairs."""
        clean_df = self.df[self.metric_columns].dropna()
        if len(clean_df) < 2 or len(self.metric_columns) < 2:
            return
        
        # Create scatter plots for all pairs
        n_metrics = len(self.metric_columns)
        fig, axes = plt.subplots(n_metrics-1, n_metrics-1, figsize=(15, 15))
        
        if n_metrics == 2:
            axes = [[axes]]
        elif n_metrics == 3:
            axes = axes.reshape(2, 2)
        
        for i in range(n_metrics-1):
            for j in range(n_metrics-1):
                if j >= i:
                    metric_x = self.metric_columns[j]
                    metric_y = self.metric_columns[i+1]
                    
                    axes[i][j].scatter(clean_df[metric_x], clean_df[metric_y], 
                                     alpha=0.6, s=1)
                    axes[i][j].set_xlabel(metric_x.upper())
                    axes[i][j].set_ylabel(metric_y.upper())
                    axes[i][j].grid(True, alpha=0.3)
                else:
                    axes[i][j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplots(self, output_path: Path):
        """Plot box plots for all metrics."""
        clean_data = []
        labels = []
        
        for metric in self.metric_columns:
            data = self.df[metric].dropna()
            if len(data) > 0:
                clean_data.append(data)
                labels.append(metric.upper())
        
        if not clean_data:
            return
        
        plt.figure(figsize=(12, 6))
        plt.boxplot(clean_data, labels=labels)
        plt.title('Metric Score Distributions')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description='Image Quality Assessment Comparison')
    parser.add_argument('--image_dir', type=str, 
                       default='koniq10k_512x384',
                       help='Directory containing images to assess')
    parser.add_argument('--output_file', type=str, 
                       default='iqa_comparison_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of images to process (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for computation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save intermediate results every N images')
    
    args = parser.parse_args()
    
    # Initialize assessor
    print("Initializing Image Quality Assessor...")
    assessor = ImageQualityAssessor(device=args.device)
    
    # Run assessment
    print(f"\nStarting assessment on {args.image_dir}")
    results_df = assessor.assess_directory(
        image_dir=args.image_dir,
        output_file=args.output_file,
        sample_size=args.sample_size,
        save_interval=args.save_interval
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    analyzer = ResultAnalyzer(results_df)
    analyzer.print_statistics()
    analyzer.correlation_analysis()
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        analyzer.create_visualizations()
    
    print(f"\nComparison completed! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
