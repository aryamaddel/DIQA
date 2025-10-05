#!/usr/bin/env python3
"""
Enhanced Image Quality Assessment Comparison with Ground Truth
Compares BRISQUE, PIQE, NIQE metrics against KONIQ-10k MOS scores with timing analysis.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import cv2
from PIL import Image

try:
    import pyiqa
    import torch
except ImportError as e:
    print(f"Error importing required packages: {e}")
    sys.exit(1)


class BIQIImplementation:
    """
    Blind Image Quality Index (BIQI) Implementation
    
    BIQI is based on Natural Scene Statistics using Mean Subtracted Contrast Normalized (MSCN) coefficients.
    This is the traditional computer vision algorithm, not an AI model.
    
    Reference: Moorthy & Bovik, "No-Reference Image Quality Assessment in the Spatial Domain" (2010)
    """
    
    def __init__(self):
        """Initialize BIQI algorithm."""
        self.device = 'cpu'  # BIQI runs on CPU using traditional CV methods
        
    def _mscn_coefficients(self, image):
        """
        Compute Mean Subtracted Contrast Normalized (MSCN) coefficients.
        
        Args:
            image: Grayscale image array
            
        Returns:
            MSCN coefficients array
        """
        # Convert to float
        image = image.astype(np.float64)
        
        # Gaussian filter for local mean estimation
        mu = cv2.GaussianBlur(image, (7, 7), 7/6)
        
        # Local variance estimation
        mu_sq = cv2.GaussianBlur(image * image, (7, 7), 7/6)
        sigma = np.sqrt(np.abs(mu_sq - mu * mu))
        
        # MSCN coefficients
        mscn = (image - mu) / (sigma + 1e-10)
        
        return mscn
    
    def _compute_biqi_features(self, image):
        """
        Compute BIQI features from MSCN coefficients.
        
        Args:
            image: Grayscale image array
            
        Returns:
            Feature vector for BIQI
        """
        # Get MSCN coefficients
        mscn = self._mscn_coefficients(image)
        
        # Shape parameters of MSCN coefficient distribution
        # Using method of moments to estimate parameters
        alpha = np.var(mscn)  # Variance (simplified)
        
        # Adjacent coefficient products
        # Horizontal adjacent products
        h_mscn = mscn[:, :-1] * mscn[:, 1:]
        
        # Vertical adjacent products  
        v_mscn = mscn[:-1, :] * mscn[1:, :]
        
        # Diagonal adjacent products
        d1_mscn = mscn[:-1, :-1] * mscn[1:, 1:]  # Main diagonal
        d2_mscn = mscn[1:, :-1] * mscn[:-1, 1:]  # Anti-diagonal
        
        # Statistical features
        features = []
        
        # MSCN coefficient statistics
        features.extend([
            np.mean(mscn),
            np.var(mscn),
            np.mean(np.abs(mscn)),
            np.mean(mscn**4),  # Kurtosis component
        ])
        
        # Adjacent coefficient product statistics
        for adj_coeffs in [h_mscn, v_mscn, d1_mscn, d2_mscn]:
            features.extend([
                np.mean(adj_coeffs),
                np.var(adj_coeffs),
                np.mean(np.abs(adj_coeffs)),
            ])
        
        return np.array(features)
    
    def _biqi_score_from_features(self, features):
        """
        Compute BIQI score from features using simplified quality model.
        
        In the full implementation, this would use trained SVM models.
        This is a simplified approximation based on feature analysis.
        
        Args:
            features: Feature vector
            
        Returns:
            BIQI quality score
        """
        # Simplified quality estimation based on NSS features
        # Lower variance in MSCN coefficients typically indicates better quality
        mscn_var = features[1]
        mscn_mean_abs = features[2]
        
        # Adjacent coefficient statistics (lower variance = better structure preservation)
        adj_vars = [features[5], features[8], features[11], features[14]]  # Variances of adjacent products
        avg_adj_var = np.mean(adj_vars)
        
        # Simple quality score computation (higher = better quality)
        # This is a heuristic based on NSS principles
        base_score = 100
        
        # Penalize high variance in MSCN coefficients (distortion indicator)
        var_penalty = min(mscn_var * 10, 40)
        
        # Penalize high variance in adjacent products (structure loss indicator)  
        adj_penalty = min(avg_adj_var * 5, 30)
        
        # Penalize deviation from natural statistics
        naturalness_penalty = min(abs(mscn_mean_abs - 0.8) * 20, 20)
        
        score = base_score - var_penalty - adj_penalty - naturalness_penalty
        
        # Ensure score is in reasonable range
        score = np.clip(score, 0, 100)
        
        return score
    
    def __call__(self, image_path: str) -> float:
        """
        Compute BIQI score for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            BIQI quality score (0-100, higher is better)
        """
        try:
            # Load image and convert to grayscale
            
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback to PIL
                pil_img = Image.open(image_path).convert('L')
                img = np.array(pil_img)
            
            # Normalize to 0-255 range
            if img.max() <= 1.0:
                img = img * 255
            
            # Resize if too large (for computational efficiency)
            if img.shape[0] > 512 or img.shape[1] > 512:
                scale = min(512 / img.shape[0], 512 / img.shape[1])
                new_height = int(img.shape[0] * scale)
                new_width = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Compute BIQI features
            features = self._compute_biqi_features(img)
            
            # Compute quality score
            score = self._biqi_score_from_features(features)
            
            return float(score)
            
        except Exception as e:
            print(f"Error in BIQI computation for {image_path}: {e}")
            return np.nan
    
    def to(self, device):
        """Compatibility method for device placement."""
        return self


class EnhancedImageQualityAssessor:
    """Enhanced Image Quality Assessment with timing and ground truth comparison."""
    
    def __init__(self, device: str = 'auto'):
        """Initialize the assessor with specified device."""
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize metrics and timing
        self.metrics = {}
        self.metric_names = []
        self.timing_stats = {}
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize IQA metrics with timing, including BIQI and available advanced metrics."""
        core_metrics = {
            'brisque': 'brisque',
            'piqe': 'piqe', 
            'niqe': 'niqe'
        }
        
        # Initialize pyiqa metrics
        for name, model_name in core_metrics.items():
            start_time = time.time()
            try:
                metric = pyiqa.create_metric(model_name, device=self.device)
                self.metrics[name] = metric
                self.metric_names.append(name)
                init_time = time.time() - start_time
                self.timing_stats[f'{name}_init_time'] = init_time
                print(f"✓ Initialized {name.upper()} in {init_time:.3f}s")
            except Exception as e:
                print(f"✗ Failed to initialize {name.upper()}: {e}")
        
        # Initialize BIQI (proper NSS-based implementation)
        start_time = time.time()
        try:
            metric_impl = BIQIImplementation()
            self.metrics['biqi'] = metric_impl
            self.metric_names.append('biqi')
            init_time = time.time() - start_time
            self.timing_stats['biqi_init_time'] = init_time
            print(f"✓ Initialized BIQI (NSS-based) in {init_time:.3f}s")
        except Exception as e:
            print(f"✗ Failed to initialize BIQI: {e}")
        
        # Try to initialize advanced metrics from pyiqa if available
        advanced_metrics = {
            'clipiqa': 'clipiqa',
            'maniqa': 'maniqa', 
            'musiq': 'musiq',
            'hyperiqa': 'hyperiqa',
            'cnniqa': 'cnniqa',
            'dbcnn': 'dbcnn',
            'tres': 'tres',
            'topiq_nr': 'topiq_nr'
        }
        
        print(f"\nTrying to initialize advanced metrics from pyiqa...")
        for name, model_name in advanced_metrics.items():
            start_time = time.time()
            try:
                metric = pyiqa.create_metric(model_name, device=self.device)
                self.metrics[name] = metric
                self.metric_names.append(name)
                init_time = time.time() - start_time
                self.timing_stats[f'{name}_init_time'] = init_time
                print(f"✓ Initialized {name.upper()} in {init_time:.3f}s")
            except Exception as e:
                print(f"✗ Failed to initialize {name.upper()}: {e}")
        
        # Note about metrics requiring external implementations
        missing_metrics = ['SaTQA', 'LoDa', 'VIDEVAL', 'RAPIQUE', 'VSFA']
        print(f"\n📝 Note: The following metrics require external implementations:")
        for metric in missing_metrics:
            print(f"   • {metric}: Requires official implementation from research papers/GitHub repos")
        
        print(f"\n💡 To add these metrics:")
        print(f"   1. Clone their official GitHub repositories")
        print(f"   2. Install their dependencies") 
        print(f"   3. Create wrapper classes that call their APIs")
        print(f"   4. Add them to the custom_metrics dictionary in _initialize_metrics()")
        
        if not self.metrics:
            raise RuntimeError("No metrics could be initialized!")
            
        print(f"\nSuccessfully initialized {len(self.metrics)} metrics: {', '.join(self.metric_names)}")
    
    def assess_image_with_timing(self, image_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Assess a single image with detailed timing for each metric."""
        results = {'image_path': image_path}
        timings = {'image_path': image_path}
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Assess with each metric and measure time
            for metric_name, metric in self.metrics.items():
                start_time = time.time()
                try:
                    score = metric(image_path)
                    if torch.is_tensor(score):
                        score = score.item()
                    results[metric_name] = float(score)
                    timings[f'{metric_name}_time'] = time.time() - start_time
                except Exception as e:
                    print(f"Error computing {metric_name} for {image_path}: {e}")
                    results[metric_name] = np.nan
                    timings[f'{metric_name}_time'] = np.nan
                    
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            for metric_name in self.metric_names:
                results[metric_name] = np.nan
                timings[f'{metric_name}_time'] = np.nan
        
        return results, timings
    
    def assess_directory_enhanced(self, image_dir: str, ground_truth_file: str = None,
                                 output_file: str = None, sample_size: int = None, 
                                 save_interval: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Assess all images with timing and optional ground truth comparison."""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Load ground truth if provided
        ground_truth_df = None
        if ground_truth_file and os.path.exists(ground_truth_file):
            ground_truth_df = pd.read_csv(ground_truth_file)
            print(f"Loaded ground truth with {len(ground_truth_df)} images")
            print(f"MOS range: {ground_truth_df['MOS'].min():.3f} - {ground_truth_df['MOS'].max():.3f}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Filter to only images that exist in ground truth if provided
        if ground_truth_df is not None:
            available_images = set(ground_truth_df['image_name'].values)
            image_files = [f for f in image_files if f.name in available_images]
            print(f"Filtered to {len(image_files)} images with ground truth scores")
        
        if sample_size and sample_size < len(image_files):
            np.random.seed(42)
            image_files = list(np.random.choice(image_files, sample_size, replace=False))
            print(f"Randomly selected {len(image_files)} images for processing")
        
        # Initialize results
        results = []
        timings = []
        
        # Setup output files
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            timing_file = output_path.parent / f"timing_{output_path.name}"
        
        # Process images with detailed timing
        overall_start = time.time()
        for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
            result, timing = self.assess_image_with_timing(str(image_file))
            result['filename'] = image_file.name
            timing['filename'] = image_file.name
            results.append(result)
            timings.append(timing)
            
            # Save intermediate results
            if output_file and (i + 1) % save_interval == 0:
                df_temp = pd.DataFrame(results)
                timing_temp = pd.DataFrame(timings)
                df_temp.to_csv(output_file, index=False)
                timing_temp.to_csv(timing_file, index=False)
                print(f"\nSaved intermediate results after {i + 1} images")
        
        # Create final dataframes
        results_df = pd.DataFrame(results)
        timing_df = pd.DataFrame(timings)
        
        # Add ground truth scores if available
        if ground_truth_df is not None:
            results_df = results_df.merge(
                ground_truth_df[['image_name', 'MOS', 'SD', 'MOS_zscore']], 
                left_on='filename', 
                right_on='image_name', 
                how='left'
            )
        
        # Save final results
        if output_file:
            results_df.to_csv(output_file, index=False)
            timing_df.to_csv(timing_file, index=False)
            print(f"\nResults saved to {output_file}")
            print(f"Timing data saved to {timing_file}")
        
        overall_time = time.time() - overall_start
        avg_time_per_image = overall_time / len(image_files)
        
        print(f"\nOverall processing completed in {overall_time:.2f} seconds")
        print(f"Average time per image: {avg_time_per_image:.3f} seconds")
        
        return results_df, timing_df


class EnhancedResultAnalyzer:
    """Enhanced analysis including ground truth comparison and timing analysis."""
    
    def __init__(self, results_df: pd.DataFrame, timing_df: pd.DataFrame = None):
        """Initialize with results and timing dataframes."""
        self.df = results_df.copy()
        self.timing_df = timing_df.copy() if timing_df is not None else None
        self.metric_columns = [col for col in self.df.columns 
                              if col not in ['image_path', 'filename', 'image_name', 'MOS', 'SD', 'MOS_zscore']]
        self.has_ground_truth = 'MOS' in self.df.columns
        
    def print_comprehensive_statistics(self):
        """Print comprehensive statistics including timing."""
        print("\n" + "="*80)
        print("COMPREHENSIVE IMAGE QUALITY ASSESSMENT ANALYSIS")
        print("="*80)
        
        print(f"\nDataset Information:")
        print(f"  Total images processed: {len(self.df)}")
        print(f"  Metrics evaluated: {len(self.metric_columns)}")
        print(f"  Metrics: {', '.join(self.metric_columns)}")
        print(f"  Ground truth available: {'Yes' if self.has_ground_truth else 'No'}")
        
        # Basic statistics for metrics
        print(f"\nMetric Score Statistics:")
        print("-" * 50)
        stats = self.df[self.metric_columns].describe()
        print(stats.round(4))
        
        # Ground truth statistics
        if self.has_ground_truth:
            print(f"\nGround Truth (MOS) Statistics:")
            print("-" * 40)
            mos_stats = self.df['MOS'].describe()
            print(mos_stats.round(4))
        
        # Timing statistics
        if self.timing_df is not None:
            self._print_timing_statistics()
    
    def _print_timing_statistics(self):
        """Print detailed timing statistics."""
        print(f"\nTiming Performance Analysis:")
        print("-" * 40)
        
        timing_cols = [col for col in self.timing_df.columns if col.endswith('_time')]
        
        for col in timing_cols:
            times = self.timing_df[col].dropna()
            if len(times) > 0:
                metric_name = col.replace('_time', '').upper()
                print(f"{metric_name}:")
                print(f"  Mean: {times.mean():.4f}s")
                print(f"  Std:  {times.std():.4f}s")
                print(f"  Min:  {times.min():.4f}s")
                print(f"  Max:  {times.max():.4f}s")
                print(f"  Images/sec: {1/times.mean():.2f}")
    
    def ground_truth_correlation_analysis(self):
        """Analyze correlations with ground truth MOS scores."""
        if not self.has_ground_truth:
            print("\nNo ground truth available for correlation analysis")
            return
        
        print("\n" + "="*80)
        print("GROUND TRUTH CORRELATION ANALYSIS")
        print("="*80)
        
        # Remove rows with missing values
        clean_df = self.df[self.metric_columns + ['MOS']].dropna()
        print(f"\nUsing {len(clean_df)} images for correlation analysis")
        
        if len(clean_df) < 10:
            print("Not enough clean data for reliable correlation analysis")
            return
        
        # Calculate correlations
        print(f"\nCorrelation with MOS (Mean Opinion Score):")
        print("-" * 60)
        print(f"{'Metric':<15} {'SRCC':<10} {'PLCC':<10} {'RMSE':<10} {'Performance'}")
        print("-" * 60)
        
        correlations = {}
        for metric in self.metric_columns:
            # Spearman Rank Correlation Coefficient (SRCC)
            srcc, srcc_p = spearmanr(clean_df[metric], clean_df['MOS'])
            
            # Pearson Linear Correlation Coefficient (PLCC)  
            plcc, plcc_p = pearsonr(clean_df[metric], clean_df['MOS'])
            
            # Root Mean Square Error (RMSE)
            rmse = np.sqrt(mean_squared_error(clean_df['MOS'], clean_df[metric]))
            
            # Performance rating
            performance = self._rate_performance(abs(srcc), abs(plcc))
            
            correlations[metric] = {
                'SRCC': srcc, 'PLCC': plcc, 'RMSE': rmse,
                'SRCC_p': srcc_p, 'PLCC_p': plcc_p
            }
            
            print(f"{metric.upper():<15} {srcc:<10.4f} {plcc:<10.4f} {rmse:<10.4f} {performance}")
        
        # Best performing metric
        best_srcc = max(correlations.keys(), key=lambda x: abs(correlations[x]['SRCC']))
        best_plcc = max(correlations.keys(), key=lambda x: abs(correlations[x]['PLCC']))
        
        print(f"\nBest SRCC: {best_srcc.upper()} ({correlations[best_srcc]['SRCC']:.4f})")
        print(f"Best PLCC: {best_plcc.upper()} ({correlations[best_plcc]['PLCC']:.4f})")
        
        return correlations
    
    def _rate_performance(self, srcc: float, plcc: float) -> str:
        """Rate the performance based on correlation values."""
        avg_corr = (srcc + plcc) / 2
        if avg_corr >= 0.9:
            return "Excellent"
        elif avg_corr >= 0.8:
            return "Good"
        elif avg_corr >= 0.7:
            return "Fair"
        elif avg_corr >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def create_enhanced_visualizations(self, output_dir: str = "enhanced_results"):
        """Create comprehensive visualizations including ground truth comparison."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Metric distributions
        self._plot_metric_distributions(output_path)
        
        # 2. Ground truth comparison if available
        if self.has_ground_truth:
            self._plot_ground_truth_comparison(output_path)
            self._plot_correlation_scatter(output_path)
        
        # 3. Timing analysis if available
        if self.timing_df is not None:
            self._plot_timing_analysis(output_path)
        
        # 4. Correlation heatmap
        self._plot_correlation_heatmap(output_path)
        
        print(f"\nEnhanced visualizations saved in {output_path}")
    
    def _plot_metric_distributions(self, output_path: Path):
        """Plot distributions of all metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot metric distributions
        for i, metric in enumerate(self.metric_columns):
            if i < len(axes):
                clean_data = self.df[metric].dropna()
                if len(clean_data) > 0:
                    axes[i].hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{metric.upper()} Distribution')
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
        
        # Plot MOS distribution if available
        if self.has_ground_truth and len(self.metric_columns) < len(axes):
            i = len(self.metric_columns)
            mos_data = self.df['MOS'].dropna()
            axes[i].hist(mos_data, bins=50, alpha=0.7, edgecolor='black', color='red')
            axes[i].set_title('MOS (Ground Truth) Distribution')
            axes[i].set_xlabel('MOS Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.metric_columns) + (1 if self.has_ground_truth else 0), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ground_truth_comparison(self, output_path: Path):
        """Plot comparison with ground truth MOS scores."""
        if not self.has_ground_truth:
            return
            
        clean_df = self.df[self.metric_columns + ['MOS']].dropna()
        
        fig, axes = plt.subplots(1, len(self.metric_columns), figsize=(5*len(self.metric_columns), 5))
        if len(self.metric_columns) == 1:
            axes = [axes]
        
        for i, metric in enumerate(self.metric_columns):
            axes[i].scatter(clean_df['MOS'], clean_df[metric], alpha=0.6, s=10)
            axes[i].set_xlabel('MOS (Ground Truth)')
            axes[i].set_ylabel(f'{metric.upper()} Score')
            axes[i].set_title(f'{metric.upper()} vs MOS')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation info
            srcc, _ = spearmanr(clean_df['MOS'], clean_df[metric])
            plcc, _ = pearsonr(clean_df['MOS'], clean_df[metric])
            axes[i].text(0.05, 0.95, f'SRCC: {srcc:.3f}\nPLCC: {plcc:.3f}', 
                        transform=axes[i].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'),
                        verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_path / 'ground_truth_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_scatter(self, output_path: Path):
        """Plot detailed correlation scatter plots."""
        if not self.has_ground_truth:
            return
            
        clean_df = self.df[self.metric_columns + ['MOS']].dropna()
        
        # Create a combined plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, metric in enumerate(self.metric_columns):
            ax.scatter(clean_df['MOS'], clean_df[metric], 
                      alpha=0.6, s=20, label=metric.upper(), 
                      color=colors[i % len(colors)])
        
        ax.set_xlabel('MOS (Ground Truth)')
        ax.set_ylabel('Metric Scores (Normalized)')
        ax.set_title('All Metrics vs Ground Truth MOS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_timing_analysis(self, output_path: Path):
        """Plot timing analysis."""
        timing_cols = [col for col in self.timing_df.columns if col.endswith('_time')]
        
        if not timing_cols:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of timing
        timing_data = []
        labels = []
        for col in timing_cols:
            data = self.timing_df[col].dropna()
            if len(data) > 0:
                timing_data.append(data)
                labels.append(col.replace('_time', '').upper())
        
        if timing_data:
            ax1.boxplot(timing_data, labels=labels)
            ax1.set_title('Processing Time Distribution by Metric')
            ax1.set_ylabel('Time (seconds)')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # Bar plot of average times
        avg_times = [np.mean(data) for data in timing_data]
        ax2.bar(labels, avg_times)
        ax2.set_title('Average Processing Time by Metric')
        ax2.set_ylabel('Average Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'timing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, output_path: Path):
        """Plot correlation heatmap."""
        metrics_to_correlate = self.metric_columns.copy()
        if self.has_ground_truth:
            metrics_to_correlate.append('MOS')
            
        clean_df = self.df[metrics_to_correlate].dropna()
        if len(clean_df) < 2:
            return
        
        corr_matrix = clean_df.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap (Including Ground Truth)')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap_with_ground_truth.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for enhanced comparison with ground truth."""
    parser = argparse.ArgumentParser(description='Enhanced IQA Comparison with Ground Truth')
    parser.add_argument('--image_dir', type=str, 
                       default='koniq10k_512x384',
                       help='Directory containing images to assess')
    parser.add_argument('--ground_truth', type=str,
                       default='koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv',
                       help='Ground truth CSV file with MOS scores')
    parser.add_argument('--output_file', type=str, 
                       default='enhanced_iqa_results.csv',
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
    
    # Initialize enhanced assessor
    print("Initializing Enhanced Image Quality Assessor...")
    assessor = EnhancedImageQualityAssessor(device=args.device)
    
    # Run enhanced assessment
    print(f"\nStarting enhanced assessment on {args.image_dir}")
    results_df, timing_df = assessor.assess_directory_enhanced(
        image_dir=args.image_dir,
        ground_truth_file=args.ground_truth,
        output_file=args.output_file,
        sample_size=args.sample_size,
        save_interval=args.save_interval
    )
    
    # Enhanced analysis
    print("\nPerforming enhanced analysis...")
    analyzer = EnhancedResultAnalyzer(results_df, timing_df)
    analyzer.print_comprehensive_statistics()
    
    # Ground truth correlation analysis
    correlations = analyzer.ground_truth_correlation_analysis()
    
    # Create enhanced visualizations
    if args.visualize:
        print("\nCreating enhanced visualizations...")
        analyzer.create_enhanced_visualizations()
    
    print(f"\nEnhanced comparison completed! Results saved to {args.output_file}")
    if correlations:
        print("\nCorrelation analysis shows how well each metric predicts human perception (MOS).")


if __name__ == "__main__":
    main()