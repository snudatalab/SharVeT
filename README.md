## SharVeT
SharVeT is an accurate and efficient parameter sharing framework, which performs similarity-based grouping to ensure accurate sharing, allocates parameters adaptively to preserve diversity within each group, and applies lightweight refinement with knowledge distillation to correct sharing-induced discrepancies.

This codebase is based on Basis_Sharing [[link]](https://github.com/TUDa-HWAI/Basis_Sharing).

### Repository layout
- `src/`
  - `train.py`: training loop and evaluation hooks
  - `model_factory.py`: builds SharVeT model and teacher, groups layers, allocates bases
  - `similarity.py`: structural and functional similarity and grouping
  - `group.py`: basis/coeff construction and replacement on the model
  - `models/`: SharVeT LLaMA/Mistral wrappers and basis/coefficient modules
  - `utils.py`: evaluation, lm-eval integration, basis allocation logic
  - `calib.py`: calibration data collection utilities
  - `prepare_data.py`: dataset preparation (wikitext, c4, slimpajama)
- `run/`: example configuration YAMLs

---

### Requirements
- Python 3.9+
- PyTorch
- Hugging Face transformers, datasets
- lm-eval (optional; used for evaluation)

---

### Quick start
1) Prepare a YAML config (see `run/test.yaml` for a working example).

2) Run training:
```bash
python -m src/train --yaml_config_file run/test.yaml
```

---

### Configuration (YAML)
See `run/test.yaml`. Top-level sections are merged into a single config object.

- `model_args` (examples)
  - `model_type`: "llama" | "mistral"
  - `model_name`: HF repo or local path
  - `k_name`, `q_name`, `v_name`, `o_name`, `up_name`, `down_name`, `gate_name`:
    submodule names
  - `num_group`: number of clusters for similarity grouping 
  - `strategy`: "similarity" (rank allocation strategy)
  - `include_bias_in_similarity`: include bias in similarity computation
  - `compression_ratio`: 0–100 (higher means more compression)
  - `minimum_basis_ratio`: lower bound fraction for per-group basis
  - `context_length`, `stride`: sequence/chunking parameters
  - `share_part`: parts to share  
  - `private_part`: parts to keep private 
  - `on_refinement`: enable lightweight discrepancy refinement

- `training args` (examples)
  - `learning_rate`, `train_epoch`, `train_batch_size`, `train_num_workers`
  - `distill_weight`, `feature_weight`, `distill_temperature`, `l2_weight`
  - `fp16`, `gradient_accumulation_steps`, `max_grad_norm`, `pin_memory`

- `calibration_args`
  - `dataset_name`: "slimpajama"
  - `build_calib`: whether to build calibration data (set "True" for the first run)
  - `calib_path`: where to save calibration artifacts
  - `dataset_cache_dir`, `calibration_size`, `calib_batch_size`
---

## Reference

If you use this code, please cite the following paper.
```bibtex
@misc{sharvet,
  title={SharVeT: Similarity-aware Parameter Sharing with \\ Vector-based Tuning for Efficient LLM Compression},
  author={Yun, Jeongin and Lee, Jaeri and Kim, Jongjin and Kim, Minjun and Song, Jinho and Kang, U},
  year={2026},
  booktitle={The 64th Annual Meeting of the Association for Computational Linguistics}
}
