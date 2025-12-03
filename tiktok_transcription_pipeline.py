"""
TikTok Video Transcription Pipeline

This script downloads TikTok videos, extracts audio, and transcribes them using Whisper.
It processes videos from a CSV file in a queue with proper error handling.
"""

import pandas as pd
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import whisper
import torch
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TikTokTranscriptionPipeline:
    """Pipeline for downloading TikTok videos and extracting transcriptions."""
    
    def __init__(
        self,
        csv_path: str,
        output_dir: str = "tiktok_data",
        audio_format: str = "mp3",
        whisper_model: str = "base",
        keep_audio: bool = False,  # Changed default to False to save storage
        device: Optional[str] = None,  # 'cuda', 'cpu', or None for auto-detect
        num_workers: int = 1,  # Number of parallel workers
        max_duration: int = 250  # Max video duration in seconds (default: 250s)
    ):
        """
        Initialize the pipeline.
        
        Args:
            csv_path: Path to CSV file with columns 'username' and 'id'
            output_dir: Directory to save downloaded files and transcriptions
            audio_format: Audio format to extract (mp3, m4a, wav)
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            keep_audio: Whether to keep audio files after transcription (default: False to save space)
            device: Device for Whisper inference ('cuda', 'cpu', or None for auto-detect)
            num_workers: Number of parallel workers for processing videos
            max_duration: Maximum video duration in seconds (skip longer videos)
        """
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.keep_audio = keep_audio
        self.num_workers = num_workers
        self.max_duration = max_duration
        self.lock = Lock()  # Thread-safe file writing
        
        # Detect and set device
        self.device = self._setup_device(device)
        
        # Create directory structure
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model with explicit device
        logger.info(f"✓ Model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        self._validate_csv()
        
        # Track progress - simplified to CSV only
        self.successful_results = []
        self.failed_results = []
        self.processed_ids = set()
        self._load_existing_results()
    
    def _setup_device(self, device: Optional[str]) -> str:
        """
        Detect and configure the device for Whisper inference.
        
        Args:
            device: User-specified device ('cuda', 'cpu', or None)
            
        Returns:
            Device string to use
        """
        if device is not None:
            # User explicitly specified device
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return 'cpu'
            logger.info(f"Using user-specified device: {device}")
            return device
        
        # Auto-detect device
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ GPU: {gpu_name}")
            return 'cuda'
        else:
            logger.info("⚠ Using CPU (slower)")
            return 'cpu'
    
    def _validate_csv(self):
        """Validate that CSV has required columns."""
        required_columns = ['username', 'id']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        logger.info(f"✓ Loaded {len(self.df)} videos")
    
    def _load_existing_results(self):
        """Load existing results to avoid reprocessing."""
        success_file = self.output_dir / "transcriptions.csv"
        
        if success_file.exists():
            df = pd.read_csv(success_file)
            self.processed_ids.update(df['video_id'].astype(str).tolist())
            # Load existing successful results
            for _, row in df.iterrows():
                self.successful_results.append(row.to_dict())
            logger.info(f"✓ Resuming: {len(df)} already processed")
    
    def _save_results(self):
        """Save current results to CSV files."""
        # Save successful transcriptions
        if self.successful_results:
            df = pd.DataFrame(self.successful_results)
            df.to_csv(self.output_dir / "transcriptions.csv", index=False)
        
        # Save failed videos
        if self.failed_results:
            df = pd.DataFrame(self.failed_results)
            df.to_csv(self.output_dir / "failed_videos.csv", index=False)
    
    def _is_already_processed(self, video_id: str) -> bool:
        """Check if video has already been processed."""
        return str(video_id) in self.processed_ids
    
    def download_audio(self, username: str, video_id: str) -> Optional[Path]:
        """
        Download audio from TikTok video using yt-dlp.
        
        Args:
            username: TikTok username
            video_id: TikTok video ID
            
        Returns:
            Path to downloaded audio file, or None if failed
        """
        video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
        audio_path = self.audio_dir / f"{video_id}.{self.audio_format}"
        
        # Skip if already downloaded
        if audio_path.exists():
            return audio_path
        
        try:
            # Add delay to avoid rate limiting
            import random
            time.sleep(random.uniform(1, 3))  # Random delay 1-3 seconds
            
            # yt-dlp command to download audio only
            command = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "--extract-audio",
                "--audio-format", self.audio_format,
                "--audio-quality", "0",
                "-o", str(audio_path.with_suffix('')),
                video_url
            ]
            
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if audio_path.exists():
                return audio_path
            else:
                logger.error(f"✗ Download failed for {video_id}: file not created")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Download timeout for {video_id}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Download failed for {video_id}: {e.stderr if e.stderr else 'unknown error'}")
            return None
        except Exception as e:
            logger.error(f"✗ Download error for {video_id}: {str(e)}")
            return None
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe."""
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            return float(result.stdout.strip())
        except:
            return 0
    
    def transcribe_audio(self, audio_path: Path, video_id: str) -> Optional[Dict]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            video_id: TikTok video ID
            
        Returns:
            Dictionary with transcription results, or None if failed
        """
        try:
            # Check duration before transcribing
            duration = self.get_audio_duration(audio_path)
            if duration > self.max_duration:
                logger.warning(f"⊘ Video too long ({duration:.0f}s > {self.max_duration}s), skipping")
                return {'skip': True, 'reason': f'Skipped: video too long ({duration:.0f}s > {self.max_duration}s limit)'}
            
            # Transcribe using Whisper with FP16 for GPU acceleration
            result = self.whisper_model.transcribe(
                str(audio_path),
                verbose=False,
                language=None,
                task="transcribe",
                fp16=(self.device == 'cuda')  # Enable FP16 for GPU
            )
            
            # Get video duration from segments
            duration = 0
            if result.get('segments'):
                duration = result['segments'][-1].get('end', 0)
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'duration': round(duration, 2)
            }
            
        except Exception as e:
            logger.error(f"✗ Transcription error for {video_id}: {str(e)}")
            return None
    
    def process_video(self, username: str, video_id: str) -> tuple[bool, Dict]:
        """
        Process a single video in queue: download → transcribe → delete.
        This processes ONE video at a time to minimize storage usage.
        
        Args:
            username: TikTok username
            video_id: TikTok video ID
            
        Returns:
            Tuple of (success: bool, result: Dict)
        """
        video_id_str = str(video_id)
        video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
        
        # Check if already processed
        if self._is_already_processed(video_id_str):
            logger.info(f"⊘ Skipped (already done)")
            return True, None  # Skip, don't add to results
        
        start_time = time.time()
        audio_path = None
        file_size_mb = 0
        
        try:
            # Step 1: Download audio (temporary)
            audio_path = self.download_audio(username, video_id)
            if not audio_path:
                return False, {
                    'video_id': video_id_str,
                    'username': username,
                    'video_url': video_url,
                    'error': 'Download failed'
                }
            
            # Get file size in MB
            file_size_mb = round(audio_path.stat().st_size / (1024 * 1024), 2)
            
            # Step 2: Transcribe audio immediately
            transcription = self.transcribe_audio(audio_path, video_id)
            if not transcription:
                return False, {
                    'video_id': video_id_str,
                    'username': username,
                    'video_url': video_url,
                    'error': 'Transcription failed'
                }
            
            # Check if video was skipped due to length
            if transcription.get('skip'):
                return False, {
                    'video_id': video_id_str,
                    'username': username,
                    'video_url': video_url,
                    'error': transcription['reason']
                }
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            # Success
            return True, {
                'video_id': video_id_str,
                'username': username,
                'video_url': video_url,
                'file_size_mb': file_size_mb,
                'video_length_sec': transcription['duration'],
                'transcription': transcription['text'],
                'language': transcription['language'],
                'processing_time_sec': processing_time
            }
            
        finally:
            # Always clean up audio file if not keeping
            if not self.keep_audio and audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass
    
    def run(self, max_videos: Optional[int] = None):
        """
        Run the pipeline on all videos in the CSV.
        
        Args:
            max_videos: Maximum number of videos to process (None for all)
        """
        videos_to_process = self.df.head(max_videos) if max_videos else self.df
        total = len(videos_to_process)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {total} videos with {self.num_workers} workers...")
        logger.info(f"{'='*60}\n")
        
        if self.num_workers == 1:
            # Sequential processing
            self._process_sequential(videos_to_process, total)
        else:
            # Parallel processing
            self._process_parallel(videos_to_process, total)
        
        # Generate summary
        self._print_summary()
    
    def _process_sequential(self, videos_to_process, total):
        """Process videos sequentially."""
        for idx, row in videos_to_process.iterrows():
            username = row['username']
            video_id = row['id']
            
            print(f"[{idx+1}/{total}] {video_id}", end=' ')
            
            success, result = self.process_video(username, video_id)
            
            if result:  # None if skipped
                if success:
                    self.successful_results.append(result)
                    self.processed_ids.add(str(video_id))
                    print("✓")
                else:
                    self.failed_results.append(result)
                    print("✗")
                
                # Save results after each video
                self._save_results()
        
        print()  # New line after progress
    
    def _process_parallel(self, videos_to_process, total):
        """Process videos in parallel using ThreadPoolExecutor."""
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(self.process_video, row['username'], row['id']): (row['username'], row['id'])
                for _, row in videos_to_process.iterrows()
            }
            
            # Process completed tasks
            for future in as_completed(future_to_video):
                username, video_id = future_to_video[future]
                completed += 1
                
                try:
                    success, result = future.result()
                    
                    if result:  # None if skipped
                        with self.lock:  # Thread-safe writing
                            if success:
                                self.successful_results.append(result)
                                self.processed_ids.add(str(video_id))
                                print(f"[{completed}/{total}] {video_id} ✓")
                            else:
                                self.failed_results.append(result)
                                print(f"[{completed}/{total}] {video_id} ✗")
                            
                            # Save results after each video
                            self._save_results()
                    else:
                        print(f"[{completed}/{total}] {video_id} ⊘")
                        
                except Exception as e:
                    print(f"[{completed}/{total}] {video_id} ✗ Error: {e}")
        
        print()  # New line after progress
    
    def _print_summary(self):
        """Print summary statistics."""
        successful = len(self.successful_results)
        failed = len(self.failed_results)
        total = successful + failed
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Success: {successful}/{total}")
        if failed > 0:
            logger.info(f"✗ Failed: {failed}")
            logger.info(f"  Check failed_videos.csv for details")
        logger.info(f"\nResults: {self.output_dir}/transcriptions.csv")
        logger.info(f"{'='*60}\n")
    
def main():
    """Main execution function."""
    
    # Configuration
    INPUT_DIR = "raw_csv"  # Directory with CSV files
    OUTPUT_DIR = "results"
    AUDIO_FORMAT = "mp3"
    WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
    KEEP_AUDIO = False  # Auto-delete audio after transcription
    MAX_VIDEOS = None  # Set to a number to test with fewer videos
    DEVICE = None  # None = auto-detect, 'cuda' = GPU, 'cpu' = CPU
    NUM_WORKERS = 1  # For GPU: use 1 worker (GPU processes faster sequentiall0y)
    MAX_DURATION = 250  # Max video duration in seconds (250s = 4.2 min)
    
    print(f"\nTikTok Transcription Pipeline")
    print(f"Model: {WHISPER_MODEL}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"{'='*60}\n")
    
    # Find all CSV files in input directory
    from pathlib import Path
    input_path = Path(INPUT_DIR)
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}/")
        return
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    # Process each CSV file
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(csv_files)}] Processing: {csv_file.name}")
        print(f"{'='*60}")
        
        # Create output directory for this file
        file_output_dir = Path(OUTPUT_DIR) / csv_file.stem
        
        # Create and run pipeline
        pipeline = TikTokTranscriptionPipeline(
            csv_path=str(csv_file),
            output_dir=str(file_output_dir),
            audio_format=AUDIO_FORMAT,
            whisper_model=WHISPER_MODEL,
            keep_audio=KEEP_AUDIO,
            device=DEVICE,
            num_workers=NUM_WORKERS,
            max_duration=MAX_DURATION
        )
        
        pipeline.run(max_videos=MAX_VIDEOS)
    
    print(f"\n{'='*60}")
    print(f"✓ All files processed!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

    