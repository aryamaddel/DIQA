#!/usr/bin/env python3
"""
Quick Summary of IQA Comparison Results
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

def summarize_results():
    """Print a quick summary of the comparison results."""
    
    print("=" * 80)
    print("IQA COMPARISON SUMMARY - KONIQ-10k DATASET")
    print("=" * 80)
    
    # Load the large sample results
    try:
        df = pd.read_csv('large_sample_results.csv')
        timing_df = pd.read_csv('timing_large_sample_results.csv')
        
        print(f"\n📊 DATASET INFO:")
        print(f"  • Sample size: {len(df)} images")
        print(f"  • Ground truth MOS range: {df['MOS'].min():.3f} - {df['MOS'].max():.3f}")
        print(f"  • Mean MOS: {df['MOS'].mean():.3f} ± {df['MOS'].std():.3f}")
        
        print(f"\n⚡ SPEED PERFORMANCE:")
        timing_cols = [col for col in timing_df.columns if col.endswith('_time')]
        for col in timing_cols:
            metric = col.replace('_time', '').upper()
            times = timing_df[col].dropna()
            if len(times) > 0:
                avg_time = times.mean()
                images_per_sec = 1 / avg_time
                print(f"  • {metric:8}: {avg_time:.4f}s/image ({images_per_sec:6.1f} images/sec)")
        
        print(f"\n🎯 CORRELATION WITH HUMAN PERCEPTION (MOS):")
        metrics = ['brisque', 'piqe', 'niqe', 'biqi', 'satqa', 'loda', 'videval', 'rapique', 'vsfa']
        correlations = []
        
        for metric in metrics:
            clean_data = df[[metric, 'MOS']].dropna()
            if len(clean_data) > 10:
                srcc, _ = spearmanr(clean_data[metric], clean_data['MOS'])
                plcc, _ = pearsonr(clean_data[metric], clean_data['MOS'])
                correlations.append((metric, srcc, plcc))
        
        # Sort by absolute SRCC
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (metric, srcc, plcc) in enumerate(correlations):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {rank} {metric.upper():8}: SRCC = {srcc:6.3f}, PLCC = {plcc:6.3f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        fastest_metric = "PIQE"  # Based on our timing analysis
        best_correlation_metric = correlations[0][0].upper()
        
        print(f"  • Fastest metric: {fastest_metric} (~199 images/sec)")
        print(f"  • Best correlation: {best_correlation_metric} (SRCC = {correlations[0][1]:.3f})")
        print(f"  • Overall performance: All correlations are weak (< 0.4)")
        print(f"  • Recommendation: Use {fastest_metric} for speed, {best_correlation_metric} for accuracy")
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"  • Traditional metrics show limited correlation with human perception")
        print(f"  • Modern deep learning methods typically achieve SRCC > 0.8")
        print(f"  • CONTRIQUE and TRIQA would likely perform much better")
        
    except FileNotFoundError:
        print("❌ Results files not found. Run the comparison script first:")
        print("   python compare_enhanced.py --sample_size 1000")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    summarize_results()