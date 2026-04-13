# diffusion_learn

这是一个和现有 `diffusion/` 主线解耦的最小实验目录。

目标不是做最好结果，而是回答最基础的问题：

1. 一个很小的 unconditional rectified flow，能不能在 4 条 motion clip 上过拟合？
2. 如果这都做不到，问题就在最核心的 diffusion/优化/数据归一化，不在音频条件、VAE、part split。
3. 如果这能做到，再逐步往上加条件和复杂结构。
4. 现在这个最小脚本已经支持 3 个关键诊断开关：
   - `--target_mode velocity|x0`
   - `--preview_mode free|teacher|both`
   - `--root_relative_first_frame`

## 推荐顺序

### 1. 先看数据

```bash
python diffusion_learn/tiny_uncond_flow.py inspect \
  --cache diffusion/cache/train_diffusion_v3_tiny32.pt \
  --clip_len 96 \
  --limit 8
```

### 2. 先做 4 条样本过拟合

```bash
python diffusion_learn/tiny_uncond_flow.py train \
  --cache diffusion/cache/train_diffusion_v3_tiny32.pt \
  --save_dir diffusion_learn/runs/tiny_overfit4 \
  --clip_len 96 \
  --overfit_n 4 \
  --epochs 200 \
  --batch_size 4 \
  --hidden_dim 128 \
  --layers 4 \
  --heads 4 \
  --lr 1e-3 \
  --preview_mode both
```

你现在不要只盯 `loss`。训练日志里会同时打印：

- `preview_free`
- `preview_teacher`

推荐这样读：

- `preview_teacher` 好，但 `preview_free` 很差：
  说明模型在 teacher-forced 下能做局部去噪，但 free rollout 采样还不稳定。
- `preview_teacher` 和 `preview_free` 都很差：
  说明训练目标或状态空间本身就有问题。
- `root_relative_first_frame` 开了之后如果 `root_err`、`root_step_ratio` 明显改善：
  说明 absolute root 是主要难点之一。
- `target_mode=x0` 比 `velocity` 明显稳：
  说明当前最难的是速度场积分，而不是去噪回归本身。

### 3. 采样看结果

```bash
python diffusion_learn/tiny_uncond_flow.py sample \
  --checkpoint diffusion_learn/runs/tiny_overfit4/best.pt \
  --cache diffusion/cache/train_diffusion_v3_tiny32.pt \
  --index 0 \
  --out_dir diffusion_learn/preview_idx0 \
  --num_steps 64 \
  --solver heun \
  --sample_mode free
```

会导出：

- `pred_motion.npy`
- `gt_motion.npy`
- `pred.bvh`
- `gt.bvh`
- `summary.json`

如果你想看 teacher-forced 单样本诊断，也可以：

```bash
python diffusion_learn/tiny_uncond_flow.py sample \
  --checkpoint diffusion_learn/runs/tiny_overfit4/best.pt \
  --cache diffusion/cache/train_diffusion_v3_tiny32.pt \
  --index 0 \
  --out_dir diffusion_learn/preview_idx0_teacher \
  --sample_mode teacher
```

## 你应该怎么学

第一阶段只回答一个问题：

- `unconditional tiny flow` 能不能记住 4 条 clip

只有这个通过之后，才值得继续加：

1. 更长 clip
2. 更多样本
3. audio 条件
4. VAE latent
5. part-aware 结构

## 为什么先做这个

你现在主线里混了太多变量：

- VAE
- part split
- root 处理
- 多路文本/音频条件
- CFG
- smoothing
- blending

这个目录的作用，就是把问题砍到只剩：

- motion normalized clip
- tiny temporal model
- rectified flow objective
- overfit
- free rollout vs teacher-forced
- velocity target vs x0 target
