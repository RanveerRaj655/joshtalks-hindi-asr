import argparse
import torch
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_DIR = Path(__file__).parent
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
TARGET_SR = 16_000


def main():
    parser = argparse.ArgumentParser(description="Baseline Whisper inference")
    parser.add_argument("--manifest", type=str,
                        default=str(BASE_DIR / "segment_manifest.csv"))
    parser.add_argument("--output", type=str,
                        default=str(BASE_DIR / "raw_asr_output.csv"))
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

        print("  BASELINE WHISPER INFERENCE")
        print(f"  Model  : {MODEL_NAME}")
    print(f"  Device : {device}")
    print(f"  Lang   : {LANGUAGE}")

    #  Load model + processor 
    print("\n[1/3] Loading model …")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    #  Load manifest 
    print("[2/3] Loading manifest …")
    df = pd.read_csv(args.manifest)
    print(f"  Found {len(df)} segments")

    # Map column names (manifest has: audio, sentence, language, duration_s)
    audio_col = "audio" if "audio" in df.columns else "audio_path"
    ref_col = "sentence" if "sentence" in df.columns else "reference_text"
    dur_col = "duration_s" if "duration_s" in df.columns else "duration"

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"  Using first {len(df)} samples")

    #  Run inference 
    print("[3/3] Running inference …\n")
    results = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Inference"):
        audio_path = Path(row[audio_col])
        reference = str(row[ref_col])
        duration = float(row[dur_col]) if dur_col in df.columns else 0.0

        # Load audio
        if not audio_path.exists():
            skipped += 1
            continue

        try:
            audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
        except Exception as e:
            skipped += 1
            continue

        # Extract features
        input_features = processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(device)

        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=225)

        # Decode
        transcription = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        results.append({
            "audio_file": str(audio_path.name),
            "reference": reference,
            "raw_asr_output": transcription,
            "duration": round(duration, 2),
        })

    #  Save output 
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"{'=' * 60}")
    print(f"  Processed : {len(results)}")
    print(f"  Skipped   : {skipped}")
    print(f"  Saved to  : {args.output}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
