"""
Transcribe TikTok videos from Parquet files

Processes parquet files, transcribes videos with missing voice_to_text,
and outputs updated parquet files with all voice_to_text entries filled.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tiktok_transcription_pipeline import TikTokTranscriptionPipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def process_parquet_file(parquet_path, output_dir="transcribed_parquet", 
                        whisper_model="small", device=None, max_duration=120):
    """
    Process a single parquet file: transcribe videos with missing voice_to_text.
    
    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory to save output parquet files
        whisper_model: Whisper model to use
        device: Device for Whisper ('cuda', 'cpu', or None for auto)
        max_duration: Max video duration in seconds
    """
    
    print("\n" + "="*70)
    print(f"Processing: {parquet_path.name}")
    print("="*70)
    
    # Prefer existing output so we skip already-transcribed videos when resuming
    output_path = Path(output_dir) / parquet_path.name
    if output_path.exists():
        try:
            df = pd.read_parquet(output_path)
            print(f"✓ Loaded existing output ({len(df)} rows) - will skip already-transcribed")
        except Exception:
            df = pd.read_parquet(parquet_path)
            print(f"✓ Loaded from raw ({len(df)} rows)")
    else:
        try:
            df = pd.read_parquet(parquet_path)
            print(f"✓ Loaded from raw ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error reading parquet: {e}")
            return None
    
    # Validate columns
    required_cols = ['username', 'id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return None
    
    # Check voice_to_text column
    if 'voice_to_text' not in df.columns:
        print("  Creating voice_to_text column...")
        df['voice_to_text'] = None
    
    # Identify videos that need transcription
    # voice_to_text is missing if: null, empty string, or NaN
    mask_missing = (
        df['voice_to_text'].isna() | 
        (df['voice_to_text'].astype(str).str.strip() == '') |
        (df['voice_to_text'].astype(str).str.strip() == 'nan')
    )
    
    videos_to_transcribe = df[mask_missing].copy()
    already_have_text = df[~mask_missing].copy()
    
    print(f"\nStatus:")
    print(f"  Already have voice_to_text: {len(already_have_text)}")
    print(f"  Need transcription: {len(videos_to_transcribe)}")
    
    if len(videos_to_transcribe) == 0:
        print("  ✓ All videos already have voice_to_text!")
        # Just save the file as-is
        output_path = Path(output_dir) / parquet_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  ✓ Saved to: {output_path}")
        return df
    
    # Create unique temporary files for this parquet file
    file_id = parquet_path.stem  # e.g., "2025-10-01_audio_extraction_registry"
    temp_csv = Path(f"temp_transcribe_{file_id}.csv")
    videos_to_transcribe[['username', 'id']].to_csv(temp_csv, index=False)
    
    # Create unique temporary output directory for this file
    temp_output = Path(f"temp_transcribe_output_{file_id}")
    temp_output.mkdir(exist_ok=True)
    
    print(f"\nTranscribing {len(videos_to_transcribe)} videos...")
    
    # Run transcription pipeline
    try:
        pipeline = TikTokTranscriptionPipeline(
            csv_path=str(temp_csv),
            output_dir=str(temp_output),
            whisper_model=whisper_model,
            keep_audio=False,
            device=device,
            num_workers=1,
            max_duration=max_duration
        )
        
        pipeline.run()
        
        # Load transcription results
        results_file = temp_output / "transcriptions.csv"
        
        if results_file.exists():
            results_df = pd.read_csv(results_file)
            print(f"  ✓ Successfully transcribed: {len(results_df)} videos")
            
            # Create mapping: video_id -> transcription
            transcription_map = dict(zip(
                results_df['video_id'].astype(str),
                results_df['transcription']
            ))
            
            # Update voice_to_text in original dataframe
            def get_transcription(row):
                video_id = str(row['id'])
                if video_id in transcription_map:
                    return transcription_map[video_id]
                return row.get('voice_to_text', None)
            
            # Update only rows that needed transcription
            df.loc[mask_missing, 'voice_to_text'] = df[mask_missing].apply(
                get_transcription, axis=1
            )
            
            # Count how many were successfully transcribed
            new_transcriptions = df[mask_missing]['voice_to_text'].notna().sum()
            print(f"  ✓ Updated {new_transcriptions} voice_to_text entries")
            
        else:
            print("  ✗ No transcription results found")
            
    except Exception as e:
        print(f"  ✗ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temp files for this specific parquet file
        if temp_csv.exists():
            temp_csv.unlink()
        if temp_output.exists():
            import shutil
            shutil.rmtree(temp_output)
    
    # Save updated parquet file
    output_path = Path(output_dir) / parquet_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Saved updated parquet to: {output_path}")
        
        # Summary
        final_missing = df['voice_to_text'].isna().sum()
        final_total = len(df)
        print(f"\nFinal Status:")
        print(f"  Total videos: {final_total}")
        print(f"  With voice_to_text: {final_total - final_missing}")
        print(f"  Still missing: {final_missing}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error saving parquet: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe TikTok videos from parquet files')
    parser.add_argument('--input-dir', default='raw_parquet', help='Directory with parquet files')
    parser.add_argument('--output-dir', default='transcribed_parquet', help='Output directory')
    parser.add_argument('--whisper-model', default='small', help='Whisper model (tiny|base|small|medium|large)')
    parser.add_argument('--device', default=None, help='Device: None (auto), cuda, or cpu')
    parser.add_argument('--max-duration', type=int, default=180, help='Max video duration in seconds (2 min = 120s)')
    parser.add_argument('--num-files', type=int, help='Process only first N files (for testing)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TIKTOK PARQUET TRANSCRIPTION PIPELINE")
    print("="*70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Whisper model: {args.whisper_model}")
    print(f"Max duration: {args.max_duration}s")
    print("="*70)
    
    # Find parquet files
    input_path = Path(args.input_dir)
    parquet_files = sorted(input_path.glob("*.parquet"))
    
    if args.num_files:
        parquet_files = parquet_files[:args.num_files]
    
    if not parquet_files:
        print(f"✗ No parquet files found in {args.input_dir}/")
        return
    
    print(f"\nFound {len(parquet_files)} parquet files to process\n")
    
    # Cleanup any leftover temp directories from previous runs
    import shutil
    for temp_dir in Path('.').glob('temp_transcribe_output_*'):
        if temp_dir.is_dir():
            shutil.rmtree(temp_dir)
    for temp_csv in Path('.').glob('temp_transcribe_*.csv'):
        if temp_csv.is_file():
            temp_csv.unlink()
    
    # Process each file
    successful = 0
    failed = 0
    
    for idx, parquet_file in enumerate(parquet_files, 1):
        print(f"\n[{idx}/{len(parquet_files)}] {parquet_file.name}")
        
        result = process_parquet_file(
            parquet_file,
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            device=args.device,
            max_duration=args.max_duration
        )
        
        if result is not None:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Successful: {successful}/{len(parquet_files)}")
    print(f"Failed: {failed}/{len(parquet_files)}")
    print(f"\nOutput files saved to: {args.output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
