import os
import re
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, Audio

#  Configuration 
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "FT Data - data.csv"
RAW_AUDIO_DIR = BASE_DIR / "raw_audio"
PROCESSED_AUDIO_DIR = BASE_DIR / "processed_audio"
OUTPUT_DATASET_DIR = BASE_DIR / "hindi_asr_dataset"

TARGET_SR = 16_000          # Whisper expects 16 kHz
MIN_DURATION = 1.0          # seconds – skip shorter segments
MAX_DURATION = 30.0         # seconds – skip longer segments
TOP_DB_TRIM = 20            # dB threshold for silence trimming
MAX_DOWNLOAD_WORKERS = 4    # parallel download threads
RETRY_COUNT = 3             # retries per download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("asr-pipeline")


#  Helpers 

def _download_with_retry(url: str, dest: Path, retries: int = RETRY_COUNT) -> bool:
    """Download a file from *url* to *dest* with retries. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif resp.status_code == 404:
                return False  # no point retrying a 404
        except requests.RequestException:
            pass
        if attempt < retries:
            time.sleep(2 ** attempt)
    return False


def _try_fallback_url(original_url: str) -> str | None:
    """
    The CSV URLs use the `joshtalks-data-collection` bucket which may 404.
    Try mapping to the `upload_goai` bucket as a fallback.
    Pattern: .../joshtalks-data-collection/hq_data/hi/{folder_id}/{file}
           → .../upload_goai/{folder_id}/{file}
    """
    match = re.search(
        r"joshtalks-data-collection/hq_data/hi/(\d+)/(.+)", original_url
    )
    if match:
        folder_id, filename = match.group(1), match.group(2)
        return f"https://storage.googleapis.com/upload_goai/{folder_id}/{filename}"
    return None


def _smart_download(url: str, dest: Path) -> bool:
    """Try original URL, then fallback URL."""
    if dest.exists():
        return True
    if _download_with_retry(url, dest):
        return True
    fallback = _try_fallback_url(url)
    if fallback and _download_with_retry(fallback, dest):
        return True
    return False


#  Stage 1: Download 

def download_all(df: pd.DataFrame):
    """Download all audio files and transcription JSONs in parallel."""
    log.info("Stage 1 ▸ Downloading audio & transcription files …")

    tasks = []  # (url, dest_path, label)
    for _, row in df.iterrows():
        rid = row["recording_id"]
        # Audio
        audio_dest = RAW_AUDIO_DIR / f"{rid}_audio.wav"
        tasks.append((row["rec_url_gcp"], audio_dest, f"audio-{rid}"))
        # Transcription
        trans_dest = RAW_AUDIO_DIR / f"{rid}_transcription.json"
        tasks.append((row["transcription_url_gcp"], trans_dest, f"trans-{rid}"))

    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_smart_download, url, dest): label
            for url, dest, label in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            if future.result():
                success += 1
            else:
                fail += 1
                log.warning("  ✗ Failed: %s", label)

    log.info("  Downloads complete — %d ok, %d failed", success, fail)


#  Stage 2: Process segments 

def _process_one_recording(rid: int) -> list[dict]:
    """
    For a single recording:
      1. Load transcription segments
      2. Load full audio
      3. For each segment: slice, resample, trim, normalise, filter by duration
      4. Save processed .wav and return metadata rows
    """
    trans_path = RAW_AUDIO_DIR / f"{rid}_transcription.json"
    audio_path = RAW_AUDIO_DIR / f"{rid}_audio.wav"

    if not trans_path.exists() or not audio_path.exists():
        return []

    # Load transcription segments
    with open(trans_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        segments = [segments]

    # Load full audio at native SR (will resample segments individually)
    try:
        full_audio, native_sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as e:
        log.warning("  ✗ Cannot load audio %s: %s", audio_path.name, e)
        return []

    rows = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        start_sec = float(seg.get("start", 0))
        end_sec = float(seg.get("end", 0))
        seg_duration = end_sec - start_sec

        #  Duration gate (pre-processing) 
        if seg_duration < MIN_DURATION or seg_duration > MAX_DURATION:
            continue

        #  Slice segment from full audio 
        start_sample = int(start_sec * native_sr)
        end_sample = int(end_sec * native_sr)
        segment_audio = full_audio[start_sample:end_sample]

        if len(segment_audio) == 0:
            continue

        #  Resample to 16 kHz 
        if native_sr != TARGET_SR:
            segment_audio = librosa.resample(
                segment_audio, orig_sr=native_sr, target_sr=TARGET_SR
            )

        #  Trim silence 
        segment_audio, _ = librosa.effects.trim(segment_audio, top_db=TOP_DB_TRIM)

        #  Post-trim duration check 
        post_trim_dur = len(segment_audio) / TARGET_SR
        if post_trim_dur < MIN_DURATION or post_trim_dur > MAX_DURATION:
            continue

        #  Volume normalisation (peak normalisation) 
        peak = np.max(np.abs(segment_audio))
        if peak > 0:
            segment_audio = segment_audio / peak * 0.95  # small headroom

        #  Save processed segment 
        out_name = f"{rid}_seg{idx:04d}.wav"
        out_path = PROCESSED_AUDIO_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), segment_audio, TARGET_SR)

        rows.append({
            "audio": str(out_path),
            "sentence": text,
            "language": "hi",
            "duration_s": round(post_trim_dur, 2),
        })

    return rows


def process_all(df: pd.DataFrame) -> pd.DataFrame:
    """Process every downloaded recording into segments."""
    log.info("Stage 2 ▸ Splitting & preprocessing audio segments …")

    all_rows = []
    unique_rids = df["recording_id"].unique()
    for i, rid in enumerate(unique_rids, 1):
        rows = _process_one_recording(rid)
        all_rows.extend(rows)
        if i % 10 == 0 or i == len(unique_rids):
            log.info("  Processed %d / %d recordings  (%d segments so far)",
                     i, len(unique_rids), len(all_rows))

    log.info("  Processing complete — %d valid segments", len(all_rows))
    return pd.DataFrame(all_rows)


#  Stage 3: Build HuggingFace Dataset 

def build_hf_dataset(seg_df: pd.DataFrame):
    """Convert processed segments into a HuggingFace Dataset and save to disk."""
    log.info("Stage 3 ▸ Building HuggingFace Dataset …")

    ds = Dataset.from_dict({
        "audio": seg_df["audio"].tolist(),
        "sentence": seg_df["sentence"].tolist(),
        "language": seg_df["language"].tolist(),
    })

    # Cast the audio column so HF loads .wav files automatically
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    ds.save_to_disk(str(OUTPUT_DATASET_DIR))
    log.info("  Dataset saved to: %s", OUTPUT_DATASET_DIR)
    return ds


#  Stage 4: Statistics 

def print_stats(seg_df: pd.DataFrame):
    """Print dataset statistics."""
    total_secs = seg_df["duration_s"].sum()
    avg_dur = seg_df["duration_s"].mean()
    median_dur = seg_df["duration_s"].median()

    print("\n" + "=" * 60)
    print("  HINDI ASR DATASET — STATISTICS")
        print(f"  Total samples    : {len(seg_df):,}")
    print(f"  Total duration   : {total_secs / 3600:.2f} hours  ({total_secs:.0f} s)")
    print(f"  Average duration : {avg_dur:.2f} s")
    print(f"  Median duration  : {median_dur:.2f} s")
    print(f"  Min duration     : {seg_df['duration_s'].min():.2f} s")
    print(f"  Max duration     : {seg_df['duration_s'].max():.2f} s")
    print(f"  Language         : hi (Hindi)")
    print(f"  Sample rate      : {TARGET_SR} Hz  (mono)")
    print(f"  Output dir       : {OUTPUT_DATASET_DIR}")
    print("=" * 60 + "\n")


#  Main 

def main():
    log.info("Loading CSV from: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.info("  Found %d recordings", len(df))

    # Create output directories
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1 – Download
    download_all(df)

    # Stage 2 – Process
    seg_df = process_all(df)

    if seg_df.empty:
        log.error("No valid segments produced. Check downloads and audio files.")
        return

    # Stage 3 – HuggingFace Dataset
    build_hf_dataset(seg_df)

    # Stage 4 – Stats
    print_stats(seg_df)

    # Also save the segment manifest as CSV for reference
    manifest_path = BASE_DIR / "segment_manifest.csv"
    seg_df.to_csv(manifest_path, index=False)
    log.info("Segment manifest saved to: %s", manifest_path)


if __name__ == "__main__":
    main()
