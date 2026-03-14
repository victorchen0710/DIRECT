# StageB Semantic Gesture (Minimal)

## Design summary

Inspired by LoM:
- Keep a strict two-stage pipeline: StageA learns the motion codec, StageB predicts StageA discrete codes.
- Reuse the current project's part-wise tokenizer layout, so the first version predicts `upper / hand / lower` codes with a shared trunk and separate heads.
- Treat StageA as frozen by default and decode predicted codes back to motion through the existing StageA decoder.

Inspired by EMAGE:
- Split rhythm and content conditions instead of feeding audio alone.
- Build rhythm features from prosody (`log_energy`, `delta_energy`, `onset_strength`, `voiced`, `log_f0`, `silence`, `speaking`).
- Build content features from TextGrid words plus boundary-aware scalar features, then fuse them through an explicit gate.
- Add motion-space auxiliary supervision by soft-decoding StageA code logits back to motion for recon / velocity / acceleration losses.
- Add masked code-hint training and code-usage regularization so StageB learns a stronger gesture prior instead of collapsing to a few frequent codes.
- Add contextual word semantics from the local `models/bert` encoder, cached offline at token level and fused with the existing TextGrid word-id and boundary features.

Simplified for this repo:
- No LoM multimodal LM pretraining, no HuBERT/T5 stack, and no EMAGE SMPL-X / FLAME / BEAT2 dependency.
- The minimum loop is `wav + TextGrid -> token predictor -> StageA decoder -> BVH`.
- Add a LoM-lite spatial motion pretraining stage with one shared trunk plus task embeddings, instead of training separate part-to-part models.
- Root/global prediction is not the main path in v1; inference uses a configurable root fallback (`zero`, `ref`, or `hold`).

## Minimal workflow

Build cache:
```bash
python stageB/cache_stage2_semantic.py \
  --manifest manifests/train.jsonl \
  --output cache/stage2_train_semantic.pt \
  --block_size 128 \
  --window_hop 128 \
  --device cuda
```

If you want the stronger text branch, rebuild the cache with local BERT enabled (default) so each token also carries contextual text features from `models/bert`:
```bash
python stageB/cache_stage2_semantic.py \
  --manifest manifests/train.jsonl \
  --output cache/stage2_train_semantic_bert.pt \
  --block_size 128 \
  --window_hop 128 \
  --device cuda \
  --bert_device cuda
```

Train prosody-only baseline:
```bash
python stageB/train_stage2_semantic.py \
  --train_cache cache/stage2_train_semantic.pt \
  --val_cache cache/stage2_val_semantic.pt \
  --save checkpoints/stage2_semantic_prosody.pt \
  --cond_mode prosody_only \
  --batch_size 8 \
  --num_workers 0 \
  --device cuda
```

Train prosody + TextGrid model:
```bash
python stageB/train_stage2_semantic.py \
  --train_cache cache/stage2_train_semantic.pt \
  --val_cache cache/stage2_val_semantic.pt \
  --save checkpoints/stage2_semantic_audio_text.pt \
  --cond_mode audio_text \
  --batch_size 8 \
  --num_workers 0 \
  --device cuda
```

The updated trainer enables AMP by default on CUDA, trains with a dual-branch objective by default, uses masked code hints unless `--disable_code_hints` is set, and supports a simple multi-GPU wrapper:
```bash
python stageB/train_stage2_semantic.py \
  --train_cache cache/stage2_train_semantic_bert.pt \
  --val_cache cache/stage2_val_semantic_bert.pt \
  --save checkpoints/stage2_semantic_audio_text.pt \
  --cond_mode audio_text \
  --batch_size 16 \
  --device cuda \
  --data_parallel \
  --gpu_ids 0,1
```

Dual-branch means:
- `generation branch`: all-mask code inputs, matched to validation/inference
- `hint branch`: partial GT code hints, auxiliary gesture-prior training

Current Stage2 semantic defaults prioritize upper-body expressiveness:
- `upper_part_w=1.0`, `hand_part_w=1.0`, `lower_part_w=0.25`
- `hint_branch_w=0.1`
- `masked_ce_w=0.2`
- `recon_w=0.15`, `vel_w=0.03`, `acc_w=0.01`
- `code_hint_batch_drop_prob=0.4`
- when loading spatial pretrain, `warmup_freeze_temporal_epochs=2`
- best checkpoint selection defaults to `semantic_upper_hand`, not raw total loss

Spatial pretraining:
```bash
python stageB/train_stage2_semantic.py \
  --train_mode spatial_pretrain \
  --train_cache cache/stage2_train_semantic.pt \
  --val_cache cache/stage2_val_semantic.pt \
  --save checkpoints/stage2_spatial_pretrain.pt \
  --batch_size 16 \
  --num_workers 0 \
  --device cuda
```

Supported LoM-lite spatial tasks:
- `upper_to_hand`
- `upper_to_lower`
- `hand_to_upper`
- `upper_hand_to_lower`

Finetune StageB from the shared spatial backbone:
```bash
python stageB/train_stage2_semantic.py \
  --train_mode stage2_finetune \
  --load_spatial_pretrain checkpoints/stage2_spatial_pretrain_best.pt \
  --train_cache cache/stage2_train_semantic_bert.pt \
  --val_cache cache/stage2_val_semantic_bert.pt \
  --save checkpoints/stage2_semantic_audio_text_spatial.pt \
  --cond_mode audio_text \
  --batch_size 16 \
  --num_workers 0 \
  --device cuda
```

Spatial sanity check:
```bash
python stageB/validate_spatial_pretrain.py \
  --cache cache/stage2_val_semantic.pt \
  --spatial_ckpt checkpoints/stage2_spatial_pretrain_best.pt \
  --task upper_to_hand \
  --sample_idx 0 \
  --out_dir outputs/stageB_spatial_check \
  --device cuda
```

Infer BVH:
```bash
python stageB/infer_stage2_semantic_bvh.py \
  --wav beat/beat_english_v0.2.1/5/5_stewart_0_8_8.wav \
  --textgrid beat/beat_english_v0.2.1/5/5_stewart_0_8_8.TextGrid \
  --stage2_ckpt checkpoints/stage2_semantic_audio_text_best.pt \
  --out_dir outputs/stage2_demo \
  --device cuda
```

Current inference defaults are:
- `root_mode=hold`
- `refine_iters=1`
- `semantic_topk=8`
- `semantic_temperature=1.1`
