"""
TikTok Video Duration Analysis

Fetches video durations without downloading and analyzes them to determine
the optimal MAX_DURATION setting for the transcription pipeline.
"""

import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")


def get_video_duration(username, video_id):
    """Get video duration using yt-dlp metadata (no download)."""
    video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
    
    try:
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-warnings', '--quiet', video_url],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            return metadata.get('duration', 0)
        return None
    except:
        return None


def fetch_durations_from_csv(input_dir="raw_csv", max_videos=None, max_total=None):
    """Fetch video durations from CSV files without downloading.
    
    Args:
        input_dir: Directory with CSV files
        max_videos: Max videos per CSV file
        max_total: Max total durations to collect (stops when reached)
    """
    
    print("\n" + "="*70)
    print("FETCHING VIDEO DURATIONS (NO DOWNLOAD)")
    print("="*70)
    
    # Find CSV files
    csv_files = sorted(Path(input_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"✗ No CSV files found in {input_dir}/")
        return []
    
    print(f"Found {len(csv_files)} CSV files")
    if max_total:
        print(f"Target: {max_total} total durations")
    
    all_durations = []
    
    for csv_file in csv_files:
        # Stop if we've reached the target
        if max_total and len(all_durations) >= max_total:
            print(f"\n✓ Reached target of {max_total} durations, stopping")
            break
        
        df = pd.read_csv(csv_file)
        
        if 'username' not in df.columns or 'id' not in df.columns:
            print(f"✗ Skipping {csv_file.name}: missing columns")
            continue
        
        # Limit videos per file
        if max_videos:
            df = df.head(max_videos)
        
        # Limit to remaining needed durations
        if max_total:
            remaining = max_total - len(all_durations)
            df = df.head(remaining)
        
        print(f"\nProcessing {csv_file.name} ({len(df)} videos)...")
        
        # Parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(get_video_duration, row['username'], row['id']): i
                for i, row in df.iterrows()
            }
            
            with tqdm(total=len(futures), desc=f"  Fetching") as pbar:
                for future in as_completed(futures):
                    duration = future.result()
                    if duration:
                        all_durations.append(duration)
                    pbar.update(1)
                    
                    # Early stop if reached target
                    if max_total and len(all_durations) >= max_total:
                        break
        
        print(f"  ✓ Total collected: {len(all_durations)} durations")
    
    return np.array(all_durations)


def analyze_and_visualize(durations):
    """Analyze durations and create visualizations."""
    
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    # Statistics
    print(f"\nTotal videos: {len(durations)}")
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {np.mean(durations):.1f}s ({np.mean(durations)/60:.1f} min)")
    print(f"  Median: {np.median(durations):.1f}s ({np.median(durations)/60:.1f} min)")
    print(f"  Std:    {np.std(durations):.1f}s")
    print(f"  Min:    {np.min(durations):.1f}s")
    print(f"  Max:    {np.max(durations):.1f}s ({np.max(durations)/60:.1f} min)")
    
    # Percentiles
    print(f"\nPercentile Analysis:")
    percentiles = [50, 75, 85, 90, 95, 98, 99]
    for p in percentiles:
        value = np.percentile(durations, p)
        count = np.sum(durations <= value)
        print(f"  {p:2d}%: {value:6.0f}s ({value/60:5.1f} min) - captures {count:5d} videos ({count/len(durations)*100:.1f}%)")
    
    # Recommendations
    print(f"\n" + "="*70)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*70)
    
    thresholds = [
        ("Conservative (90%)", np.percentile(durations, 90)),
        ("Balanced (95%)", np.percentile(durations, 95)),
        ("Inclusive (98%)", np.percentile(durations, 98)),
    ]
    
    for name, threshold in thresholds:
        captured = np.sum(durations <= threshold)
        percentage = (captured / len(durations)) * 100
        print(f"\n{name}:")
        print(f"  MAX_DURATION = {int(threshold)}  # {threshold/60:.1f} min")
        print(f"  Captures: {captured}/{len(durations)} ({percentage:.2f}%)")
        print(f"  Skips: {len(durations)-captured} ({100-percentage:.2f}%)")
    
    # Recommended
    recommended = int(np.percentile(durations, 95))
    
    print(f"\n" + "="*70)
    print("💡 RECOMMENDED SETTING:")
    print(f"   MAX_DURATION = {recommended}  # Captures 95% of videos")
    print("="*70)
    
    # Visualizations
    create_visualizations(durations, recommended)
    
    # Scenario table
    create_scenario_table(durations)
    
    return recommended


def create_visualizations(durations, recommended):
    """Create 4-panel visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Video Duration Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    ax1 = axes[0, 0]
    sns.histplot(durations, bins=50, kde=True, ax=ax1, color='skyblue', edgecolor='black')
    ax1.axvline(recommended, color='red', linestyle='--', linewidth=2, 
                label=f'Recommended: {recommended}s ({recommended/60:.1f}m)')
    ax1.axvline(np.median(durations), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(durations):.0f}s')
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Video Durations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    sns.boxplot(y=durations, ax=ax2, color='lightcoral')
    ax2.axhline(recommended, color='red', linestyle='--', linewidth=2,
                label=f'Recommended: {recommended}s')
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_title('Box Plot - Outlier Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative Distribution
    ax3 = axes[1, 0]
    sorted_dur = np.sort(durations)
    cumulative = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur) * 100
    ax3.plot(sorted_dur, cumulative, linewidth=2, color='darkblue')
    ax3.axvline(recommended, color='red', linestyle='--', linewidth=2,
                label=f'95% at {recommended}s')
    ax3.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Duration (seconds)')
    ax3.set_ylabel('Cumulative Percentage (%)')
    ax3.set_title('Cumulative Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, np.percentile(durations, 99))
    
    # 4. Percentile bars
    ax4 = axes[1, 1]
    percentiles = [50, 75, 85, 90, 95, 98, 99]
    percentile_values = [np.percentile(durations, p) for p in percentiles]
    colors = ['green' if p <= 95 else 'orange' if p <= 98 else 'red' for p in percentiles]
    
    bars = ax4.barh([f'{p}%' for p in percentiles], percentile_values, color=colors, edgecolor='black')
    ax4.axvline(recommended, color='red', linestyle='--', linewidth=2, label='Recommended')
    ax4.set_xlabel('Duration (seconds)')
    ax4.set_ylabel('Percentile')
    ax4.set_title('Duration by Percentile')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, percentile_values):
        ax4.text(val + 10, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}s ({val/60:.1f}m)', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('duration_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: duration_analysis.png")


def create_scenario_table(durations):
    """Create comparison table for different thresholds."""
    
    thresholds = [180, 300, 600, 900, 1200, 1800]
    
    data = []
    for threshold in thresholds:
        captured = np.sum(durations <= threshold)
        percentage = (captured / len(durations)) * 100
        
        data.append({
            'Threshold (sec)': threshold,
            'Threshold (min)': threshold / 60,
            'Videos Captured': captured,
            'Percentage': f'{percentage:.2f}%',
            'Videos Lost': len(durations) - captured
        })
    
    df = pd.DataFrame(data)
    df.to_csv('threshold_scenarios.csv', index=False)
    print(f"✓ Saved scenarios: threshold_scenarios.csv")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video durations for optimal MAX_DURATION')
    parser.add_argument('--input-dir', default='raw_csv', help='Directory with CSV files')
    parser.add_argument('--max-videos', type=int, help='Max videos per CSV file')
    parser.add_argument('--max-total', type=int, help='Stop after collecting N total durations')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TIKTOK VIDEO DURATION ANALYSIS")
    print("="*70)
    
    # Fetch durations
    durations = fetch_durations_from_csv(
        input_dir=args.input_dir,
        max_videos=args.max_videos,
        max_total=args.max_total
    )
    
    if len(durations) == 0:
        print("\n✗ No durations collected!")
        return
    
    # Save raw data
    pd.DataFrame({'duration': durations}).to_csv('video_durations.csv', index=False)
    print(f"\n✓ Saved raw data: video_durations.csv")
    
    # Analyze and visualize
    recommended = analyze_and_visualize(durations)
    
    print(f"\n{'='*70}")
    print("COMPLETE! Update your pipeline:")
    print(f"  Edit tiktok_transcription_pipeline.py")
    print(f"  Set: MAX_DURATION = {recommended}")
    print("="*70)


if __name__ == "__main__":
    main()

