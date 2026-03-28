import pandas as pd
import numpy as np
import jiwer
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent


def align_to_reference(ref_text: str, hyp_text: str) -> list[str]:
    """
    Align a hypothesis string to a reference string.
    Returns a list of length `len(ref_words)`, containing the word the 
    hypothesis proposed at that position, or empty string if deleted.
    """
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    
    if not ref_words:
        return []
    if not hyp_words:
        return [""] * len(ref_words)
        
    out = jiwer.process_words(ref_text, hyp_text)
    aligned = [""] * len(ref_words)
    
    for chunk in out.alignments[0]:
        if chunk.type in ('equal', 'substitute'):
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_i = chunk.ref_start_idx + k
                hyp_i = chunk.hyp_start_idx + k
                if hyp_i < len(hyp_words) and ref_i < len(aligned):
                    aligned[ref_i] = hyp_words[hyp_i]
        elif chunk.type == 'delete':
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_i = chunk.ref_start_idx + k
                if ref_i < len(aligned):
                    aligned[ref_i] = ""
                    
    return aligned


def build_lattice(ref_text: str, models_texts: list[str]) -> list[dict]:
    """
    Build a lattice bin for each position in the human reference.
    """
    ref_words = ref_text.split()
    n = len(ref_words)
    
    # Align all models to reference
    aligned_models = []
    for m_text in models_texts:
        aligned_models.append(align_to_reference(ref_text, m_text))
        
    lattice = []
    
    for i in range(n):
        ref_w = ref_words[i]
        model_words_at_i = [m[i] for m in aligned_models if m[i] != ""]
        
        # Determine consensus (words agreed upon by 3+ models)
        counts = Counter(model_words_at_i)
        valid_alternatives = set([ref_w])
        primary_alternatives = []
        
        for w, c in counts.items():
            if c >= 3 and w != ref_w:
                valid_alternatives.add(w)
                primary_alternatives.append(w)
                
        lattice.append({
            "human_ref": ref_w,
            "model_words": model_words_at_i,
            "valid_words": valid_alternatives,
            "primary": primary_alternatives[0] if primary_alternatives else ref_w
        })
        
    return lattice


def compute_lattice_wer(ref_text: str, hyp_text: str, lattice: list[dict]) -> tuple[float, float, int]:
    """
    Compute Standard WER and Lattice WER.
    Standard WER = standard edit distance.
    Lattice WER = if a substitution matches a valid lattice word, it is forgiven.
    Returns (standard_wer, lattice_wer, base_words_count).
    """
    # Clean up empty strings just in case
    ref_text = " ".join(ref_text.split())
    hyp_text = " ".join(hyp_text.split())
    
    if not ref_text:
        return 0.0, 0.0, 0
        
    out = jiwer.process_words(ref_text, hyp_text)
    
    # Standard errors
    substitutions = out.substitutions
    deletions = out.deletions
    insertions = out.insertions
    standard_errors = substitutions + deletions + insertions
    n_words = len(ref_text.split())
    
    standard_wer = standard_errors / n_words
    
    # Lattice errors: forgive substitutions that are in the lattice
    lattice_errors = standard_errors
    hyp_words = hyp_text.split()
    
    for chunk in out.alignments[0]:
        if chunk.type == 'substitute':
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_i = chunk.ref_start_idx + k
                hyp_i = chunk.hyp_start_idx + k
                if hyp_i < len(hyp_words) and ref_i < len(lattice):
                    hyp_word = hyp_words[hyp_i]
                    if hyp_word in lattice[ref_i]["valid_words"]:
                        # Forgiven!
                        lattice_errors -= 1
                        
    lattice_wer = lattice_errors / n_words
    return standard_wer, lattice_wer, n_words


def main():
    input_file = BASE_DIR / "Question 4 - Task.csv"
    print(f"Loading data from {input_file.name}...")
    df = pd.read_csv(input_file).fillna("")
    
    # Identify model columns
    model_columns = [c for c in df.columns if c.startswith("Model")]
    
    print(f"  Found {len(df)} segments.")
    print(f"  Models: {', '.join(model_columns)}\n")
    
    # We will accumulate errors to compute overall WER
    # {model_name: {"std_err": 0, "lat_err": 0, "words": 0}}
    metrics = {col: {"std_err": 0, "lat_err": 0, "words": 0} for col in model_columns}
    
    # 2. Process rows
    print("Constructing Lattices & Computing WER...")
    
    for idx, row in df.iterrows():
        ref_text = str(row["Human"])
        models_texts = [str(row[m]) for m in model_columns]
        
        # Build lattice for this segment
        lattice = build_lattice(ref_text, models_texts)
        
        # Print first 3 lattices as examples
        if idx < 3:
            print(f"\n  --- LATTICE EXAMPLE {idx+1} ---")
            print(f"  Original Ref : {ref_text}")
            for pos, bin_info in enumerate(lattice):
                valid = list(bin_info['valid_words'])
                print(f"    Pos {pos:>2} | Ref: {bin_info['human_ref']:<10} | Primary: {bin_info['primary']:<12} | Valid: {valid}")
        
        # Compute WER for each model
        num_words = len(ref_text.split())
        for model_col, m_text in zip(model_columns, models_texts):
            std_wer, lat_wer, _ = compute_lattice_wer(ref_text, m_text, lattice)
            
            # Accumulate errors
            metrics[model_col]["std_err"] += std_wer * num_words
            metrics[model_col]["lat_err"] += lat_wer * num_words
            metrics[model_col]["words"] += num_words

    # 4. Final Output Table
    print("\nOVERALL PERFORMANCE COMPARISON")
    
    results_list = []
    
    for model_col in model_columns:
        m_data = metrics[model_col]
        total_words = m_data["words"]
        
        overall_std = (m_data["std_err"] / total_words) * 100 if total_words > 0 else 0
        overall_lat = (m_data["lat_err"] / total_words) * 100 if total_words > 0 else 0
        delta = overall_std - overall_lat
        
        # If delta is significant (>0.5%), the model was being penalized fairly for bad human reference
        fairly_penalized = "Yes" if delta > 0.5 else "No"
        
        results_list.append({
            "Model": model_col,
            "Standard WER (%)": round(overall_std, 2),
            "Lattice WER (%)": round(overall_lat, 2),
            "Delta (%)": round(delta, 2),
            "Fairly penalized?": fairly_penalized
        })
        
    res_df = pd.DataFrame(results_list)
    print(res_df.to_string(index=False))
    
    # 5. Save results
    out_file = BASE_DIR / "lattice_wer_results.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file.name}")


if __name__ == "__main__":
    main()
