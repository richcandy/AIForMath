## Lean4 Proof Experiment

First-pass experiment setup for `formal_statement -> proof body` with a small split sized for a 4 GB GPU.

### Files

- `prepare_lean4_experiment_data.py`: builds the `1600/200/200` short-sample split.
- `generate_zero_shot_proofs.py`: runs the base model on a split and saves proof candidates.
- `evaluate_lean_passk.py`: compiles generated proofs with Lean and reports `Pass@1/Pass@k`.

### Prepare data

```powershell
python .\prepare_lean4_experiment_data.py
```

Output directory:

```text
E:\GraduationProject\experiment_data\lean4_small_1600_200_200
```

### Run zero-shot baseline

`Pass@1` example:

```powershell
python .\generate_zero_shot_proofs.py --dataset .\experiment_data\lean4_small_1600_200_200\test.jsonl --output .\runs\zero_shot_test_pass1.jsonl --k 1 --temperature 0 --max-new-tokens 256
```

`Pass@5` example:

```powershell
python .\generate_zero_shot_proofs.py --dataset .\experiment_data\lean4_small_1600_200_200\test.jsonl --output .\runs\zero_shot_test_pass5.jsonl --k 5 --temperature 0.7 --top-p 0.95 --max-new-tokens 256
```

### Evaluate with Lean

`evaluate_lean_passk.py` runs `lake env lean`, so it must point at a Lean project that already has `Mathlib` available.

Verified local project on this machine:

```text
E:\mathematics_in_lean
```

```powershell
python .\evaluate_lean_passk.py --predictions .\runs\zero_shot_test_pass1.jsonl --output .\runs\zero_shot_test_pass1_eval.jsonl --project-dir E:\mathematics_in_lean
```

The script writes per-sample results plus a summary JSON with `Pass@1`, `Pass@k`, and failure counts.
