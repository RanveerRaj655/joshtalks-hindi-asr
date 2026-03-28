import argparse
import torch
import jiwer
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

#  Configuration 

BASE_DIR = Path(__file__).parent
BASELINE_MODEL = "openai/whisper-small"
DEFAULT_FINETUNED = str(BASE_DIR / "whisper-small-hi-finetuned")
LANGUAGE = "hi"
TASK = "transcribe"
FLEURS_DATASET = "google/fleurs"
FLEURS_SPLIT = "hi_in"


#  Evaluation logic 

def evaluate_model(
    model_name_or_path: str,
    processor: WhisperProcessor,
    test_dataset,
    device: str,
    batch_size: int = 8,
    label: str = "Model",
) -> float:
    """
    Run inference on the test dataset and compute WER.
    Returns WER as a percentage.
    """
    print(f"\n{'─' * 50}")
    print(f"  Evaluating: {label}")
    print(f"  Model path: {model_name_or_path}")
    print(f"  Device    : {device}")
    print(f"  Samples   : {len(test_dataset)}")
    print(f"{'─' * 50}")

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()

    # Force language settings
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    all_predictions = []
    all_sample_wers = []
    all_references = []

    # Process in batches
    for i in tqdm(range(0, len(test_dataset), batch_size), desc=f"  {label}"):
        batch = test_dataset[i : i + batch_size]

        # Extract audio arrays and texts
        audios = batch["audio"]
        references = batch["transcription"]

        for audio, ref in zip(audios, references):
            # Feature extraction
            input_features = processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
            ).input_features.to(device)

            # Generate
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_length=225,
                )

            # Decode
            transcription = processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            all_predictions.append(transcription)
            all_references.append(ref.strip())

    # Compute per-sample and overall WER
    for ref, pred in zip(all_references, all_predictions):
        try:
            sample_wer = jiwer.wer(ref, pred) * 100
        except ValueError:
            sample_wer = 100.0
        all_sample_wers.append(sample_wer)

    wer_score = jiwer.wer(all_references, all_predictions) * 100

    #  Save predictions CSV 
    csv_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    csv_path = BASE_DIR / f"predictions_{csv_name}.csv"
    pred_df = pd.DataFrame({
        "reference": all_references,
        "prediction": all_predictions,
        "wer_percent": all_sample_wers,
    })
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Predictions saved to: {csv_path}")

    # Show a few example predictions
    print(f"\n  Sample predictions ({label}):")
    for j in range(min(3, len(all_predictions))):
        print(f"    REF : {all_references[j][:80]}")
        print(f"    PRED: {all_predictions[j][:80]}")
        print()

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return wer_score


#  Main 

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper baseline vs fine-tuned on FLEURS Hindi"
    )
    parser.add_argument(
        "--finetuned_model", type=str, default=DEFAULT_FINETUNED,
        help="Path to the fine-tuned Whisper model directory"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max number of test samples (for quick testing)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip baseline evaluation"
    )
    parser.add_argument(
        "--skip_finetuned", action="store_true",
        help="Skip fine-tuned evaluation"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

        print("  WHISPER EVALUATION — FLEURS HINDI TEST SET")
    
    #  Load FLEURS Hindi test split 
    # datasets>=4.x dropped loading-script support, so we load Hindi-only
    # parquet files directly from the Hub.
    print("\nLoading FLEURS Hindi test split …")
    fleurs = None

    try:
        # Try loading directly via HF parquet API for Hindi
        fleurs = load_dataset(
            "google/fleurs",
            "hi_in",
            split="test",
        )
        print("  Loaded via standard load_dataset")
    except (RuntimeError, ValueError) as e1:
        print(f"  Standard load failed: {e1}")
        try:
            # Fallback: load from the parquet files on the Hub directly
            parquet_url = (
                "https://huggingface.co/datasets/google/fleurs/resolve/"
                "refs%2Fconvert%2Fparquet/hi_in/test/0000.parquet"
            )
            fleurs = load_dataset("parquet", data_files=parquet_url, split="train")
            print("  Loaded via direct parquet URL")
        except Exception as e2:
            print(f"  Parquet URL failed: {e2}")
            try:
                # Last resort: use an older compatible method
                fleurs = load_dataset(
                    "google/fleurs", "hi_in", split="test",
                    revision="refs/convert/parquet",
                )
                print("  Loaded via parquet revision")
            except Exception as e3:
                print(f"  All strategies failed: {e3}")
                print("\n✗ Could not load FLEURS. Fix: pip install 'datasets<3'")
                return

    print(f"  Loaded {len(fleurs)} test samples")

    if args.max_samples:
        fleurs = fleurs.select(range(min(args.max_samples, len(fleurs))))
        print(f"  Using subset: {len(fleurs)} samples")

    #  Load processor 
    processor = WhisperProcessor.from_pretrained(
        BASELINE_MODEL, language=LANGUAGE, task=TASK
    )

    results = {}

    #  Evaluate baseline 
    if not args.skip_baseline:
        wer_baseline = evaluate_model(
            model_name_or_path=BASELINE_MODEL,
            processor=processor,
            test_dataset=fleurs,
            device=device,
            batch_size=args.batch_size,
            label="Whisper-small (baseline)",
        )
        results["Whisper-small baseline"] = wer_baseline

    #  Evaluate fine-tuned 
    if not args.skip_finetuned:
        ft_path = Path(args.finetuned_model)
        if ft_path.exists():
            # Use fine-tuned processor if available, else baseline
            ft_processor_path = ft_path if (ft_path / "tokenizer_config.json").exists() else BASELINE_MODEL
            ft_processor = WhisperProcessor.from_pretrained(
                str(ft_processor_path), language=LANGUAGE, task=TASK
            )
            wer_finetuned = evaluate_model(
                model_name_or_path=str(ft_path),
                processor=ft_processor,
                test_dataset=fleurs,
                device=device,
                batch_size=args.batch_size,
                label="Fine-tuned Whisper-small",
            )
            results["Fine-tuned Whisper-small"] = wer_finetuned
        else:
            print(f"\n⚠  Fine-tuned model not found at: {ft_path}")
            print("   Run train_whisper.py first, or pass --finetuned_model <path>")
            print("   Skipping fine-tuned evaluation.\n")

    #  Print results table 
    if results:
        print("\n" + "=" * 60)
        print("  RESULTS")
                print(f"\n  {'Model':<35} {'WER (%)':<10}")
        print(f"  {'─' * 35} {'─' * 10}")
        for model_name, wer in results.items():
            print(f"  {model_name:<35} {wer:<10.1f}")
        print()

        # Also print as Markdown table
        print("  Markdown table:")
        print("  | Model | WER (%) |")
        print("  |-------|---------|")
        for model_name, wer in results.items():
            print(f"  | {model_name} | {wer:.1f} |")
        print()

        # Improvement
        # Save summary CSV
        summary_df = pd.DataFrame([
            {"model": k, "wer_percent": v} for k, v in results.items()
        ])
        summary_path = BASE_DIR / "model_comparison.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Summary saved to: {summary_path}\n")

        if len(results) == 2:
            baseline_wer = results.get("Whisper-small baseline", 0)
            ft_wer = results.get("Fine-tuned Whisper-small", 0)
            if baseline_wer > 0:
                improvement = baseline_wer - ft_wer
                pct = (improvement / baseline_wer) * 100
                arrow = "↓" if improvement > 0 else "↑"
                print(f"  WER Change: {arrow} {abs(improvement):.1f}% ({abs(pct):.1f}% relative)")
        print("=" * 60 + "\n")
    else:
        print("\nNo models were evaluated.")


if __name__ == "__main__":
    main()
