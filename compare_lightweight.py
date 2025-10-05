#!/usr/bin/env python3
"""
Lightweight Image Quality Assessment Comparison Tool
Compares BRISQUE, PIQE, and NIQE metrics on KONIQ-10k dataset.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import pyiqa
    import torch
except ImportError as e:
    print(f"Error importing required packages: {e}")
    sys.exit(1)


class LightweightImageQualityAssessor:
    """Lightweight Image Quality Assessment using core metrics only."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize the assessor with specified device."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize only core metrics that don't require large downloads
        self.metrics = {}
        self.metric_names = []
        self._initialize_core_metrics()
        
    def _initialize_core_metrics(self):
        """Initialize core IQA metrics (BRISQUE, PIQE, NIQE)."""
        core_metrics = {
            'brisque': 'brisque',
            'piqe': 'piqe', 
            'niqe': 'niqe'
        }
        
        for name, model_name in core_metrics.items():
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
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
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
        
        # Find all image files (avoid duplicates)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        if sample_size and sample_size < len(image_files):
            np.random.seed(42)  # For reproducible sampling
            image_files = list(np.random.choice(image_files, sample_size, replace=False))
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
                print(f"\nSaved intermediate results after {i + 1} images")
        
        # Create final dataframe
        df = pd.DataFrame(results)
        
        # Save final results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nFinal results saved to {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/len(image_files):.3f} seconds")
        
        return df


def print_statistics(df: pd.DataFrame):
    """Print basic statistics for all metrics."""
    metric_columns = [col for col in df.columns if col not in ['image_path', 'filename']]
    
    print("\n" + "="*60)
    print("IMAGE QUALITY ASSESSMENT STATISTICS")
    print("="*60)
    
    print(f"\nDataset size: {len(df)} images")
    print(f"Metrics evaluated: {len(metric_columns)}")
    print(f"Metrics: {', '.join(metric_columns)}")
    
    print("\nDescriptive Statistics:")
    print("-" * 40)
    stats = df[metric_columns].describe()
    print(stats.round(4))
    
    print("\nMissing Values:")
    print("-" * 20)
    missing = df[metric_columns].isnull().sum()
    for metric, count in missing.items():
        percentage = (count / len(df)) * 100
        print(f"{metric}: {count} ({percentage:.1f}%)")


def correlation_analysis(df: pd.DataFrame):
    """Analyze correlations between metrics."""
    metric_columns = [col for col in df.columns if col not in ['image_path', 'filename']]
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Remove rows with any NaN values for correlation analysis
    clean_df = df[metric_columns].dropna()
    print(f"\nUsing {len(clean_df)} images for correlation analysis")
    
    if len(clean_df) < 2:
        print("Not enough clean data for correlation analysis")
        return
    
    corr_matrix = clean_df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(4))
    
    # Find correlations between different metrics
    print("\nPairwise correlations:")
    print("-" * 30)
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            metric1 = corr_matrix.columns[i]
            metric2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            print(f"{metric1} vs {metric2}: {corr_val:.4f}")


def create_quick_visualization(df: pd.DataFrame, output_dir: str = "results"):
    """Create basic visualizations."""
    metric_columns = [col for col in df.columns if col not in ['image_path', 'filename']]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Distribution plots
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(15, 5))
    if len(metric_columns) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metric_columns):
        clean_data = df[metric].dropna()
        if len(clean_data) > 0:
            axes[i].hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.upper()} Distribution')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap if we have multiple metrics
    if len(metric_columns) > 1:
        clean_df = df[metric_columns].dropna()
        if len(clean_df) > 1:
            corr_matrix = clean_df.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Metric Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nVisualization plots saved in {output_path}")


def main():
    """Main function to run the lightweight comparison."""
    parser = argparse.ArgumentParser(description='Lightweight Image Quality Assessment Comparison')
    parser.add_argument('--image_dir', type=str, 
                       default='koniq10k_512x384',
                       help='Directory containing images to assess')
    parser.add_argument('--output_file', type=str, 
                       default='iqa_lightweight_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of images to process (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for computation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--save_interval', type=int, default=500,
                       help='Save intermediate results every N images')
    
    args = parser.parse_args()
    
    # Initialize assessor
    print("Initializing Lightweight Image Quality Assessor...")
    assessor = LightweightImageQualityAssessor(device=args.device)
    
    # Run assessment
    print(f"\nStarting assessment on {args.image_dir}")
    results_df = assessor.assess_directory(
        image_dir=args.image_dir,
        output_file=args.output_file,
        sample_size=args.sample_size,
        save_interval=args.save_interval
    )
    
    # Analyze results
    print_statistics(results_df)
    correlation_analysis(results_df)
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        create_quick_visualization(results_df)
    
    print(f"\nComparison completed! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()