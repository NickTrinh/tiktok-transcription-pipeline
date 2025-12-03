# TikTok Transcription Pipeline

Extract speech from TikTok videos using Whisper AI.

## Setup

```bash
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

## Usage

1. Put CSV files in `raw_csv/` folder
2. Run: `python tiktok_transcription_pipeline.py`
3. Results in `results/` folder (one subfolder per CSV file)

## Configure

Edit `tiktok_transcription_pipeline.py`:
- `WHISPER_MODEL` - tiny|base|small|medium|large
- `DEVICE` - None (auto-detect) | 'cuda' (force GPU) | 'cpu' (force CPU)
- `NUM_WORKERS` - 1 for GPU | 3-5 for CPU
- `MAX_DURATION` - Skip videos longer than N seconds (default: 600 = 10 min)

### GPU vs CPU:
- **GPU (RTX 3050)**: 5-10x faster, use `small` or `medium` model, `NUM_WORKERS=1`
- **CPU**: Use `small` model, `NUM_WORKERS=3-5` for parallel processing

## Output

Each CSV file generates:
- `transcriptions.csv` - Results with file size, duration, transcript, language, processing time
- `failed_videos.csv` - Failed videos with errors

