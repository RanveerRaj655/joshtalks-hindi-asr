import os
import argparse
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from datasets import load_from_disk, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

#  Configuration 

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "hindi_asr_dataset"
OUTPUT_DIR = BASE_DIR / "whisper-small-hi-finetuned"

MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"


#  Data Collator 

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper that handles:
      - Padding input features (mel spectrograms) to max length
      - Padding labels (token IDs) and replacing pad tokens with -100
        so the loss function ignores them
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        #  Pad input features (mel spectrograms) 
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        #  Pad labels 
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it is ignored by loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If the BOS token is prepended in every example, remove it here
        # (the model will add its own during generation)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


#  Prepare dataset 

def prepare_dataset(example, processor):
    """
    Process a single example:
      1. Extract mel spectrogram from audio
      2. Tokenize the transcription text
    """
    audio = example["audio"]

    # Compute log-mel spectrogram
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    # Tokenize text
    labels = processor.tokenizer(example["sentence"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


#  WER Metric 

def make_compute_metrics(processor):
    """Return a compute_metrics function that calculates WER."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        wer = 100 * wer_metric.compute(
            predictions=pred_str, references=label_str
        )
        return {"wer": wer}

    return compute_metrics


#  Main 

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on Hindi ASR")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push trained model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HuggingFace Hub model ID (e.g. 'username/whisper-small-hi')")
    parser.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR),
                        help="Path to HuggingFace dataset on disk")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

        print("  WHISPER-SMALL HINDI FINE-TUNING")
    
    #  1. Load processor 
    print("\n[1/5] Loading processor …")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    #  2. Load and split dataset 
    print("[2/5] Loading dataset …")
    dataset = load_from_disk(args.dataset_dir)

    # If dataset is a Dataset (not DatasetDict), split it
    if not isinstance(dataset, DatasetDict):
        split = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            "train": split["train"],
            "eval": split["test"],
        })
    else:
        # Rename 'test' to 'eval' if present
        if "test" in dataset and "eval" not in dataset:
            dataset["eval"] = dataset["test"]
            del dataset["test"]

    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Eval : {len(dataset['eval'])} samples")

    #  3. Preprocess dataset 
    print("[3/5] Extracting features …")
    dataset = dataset.map(
        lambda ex: prepare_dataset(ex, processor),
        remove_columns=dataset.column_names["train"],
        num_proc=1,  # audio processing is not picklable with >1 workers
    )

    #  4. Load model 
    print("[4/5] Loading model …")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Force Hindi language and transcribe task during generation
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None  # unset; we use language/task

    # Disable cache (incompatible with gradient checkpointing)
    model.config.use_cache = False

    #  5. Training 
    print("[5/5] Starting training …\n")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2 or 1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        dataloader_num_workers=0,  # safe default for Windows
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    # Train!
    trainer.train()

    # Save the final model + processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n✓ Model saved to: {args.output_dir}")

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub …")
        trainer.push_to_hub()
        print("✓ Pushed to Hub!")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    

if __name__ == "__main__":
    main()
