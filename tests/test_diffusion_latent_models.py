import math

import torch

from diffusion.latent.models import LatentRectifiedFlowTransformer, MotionVAE


def test_motion_vae_shapes() -> None:
    model = MotionVAE(
        motion_dim=465,
        latent_dim=32,
        hidden_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        latent_stride=4,
        dropout=0.0,
    )
    motion = torch.randn(2, 17, 465)
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[1, 15:] = False
    out = model(motion, mask)
    assert out["recon_motion_norm"].shape == motion.shape
    assert out["latents"].shape == (2, math.ceil(17 / 4), 32)
    assert out["latent_mask"].shape == (2, math.ceil(17 / 4))
    assert torch.isfinite(out["recon_motion_norm"]).all()


def test_latent_flow_shapes_with_prefix_cfg_inputs() -> None:
    model = LatentRectifiedFlowTransformer(
        latent_dim=32,
        audio_dim=24,
        lexical_dim=16,
        global_text_dim=16,
        hidden_dim=64,
        num_layers=2,
        num_double_layers=1,
        token_refiner_layers=1,
        num_heads=4,
        dropout=0.0,
        local_attn_window=3,
    )
    x_t = torch.randn(2, 5, 32)
    timesteps = torch.rand(2)
    latent_mask = torch.ones(2, 5, dtype=torch.bool)
    audio = torch.randn(2, 18, 24)
    lexical = torch.randn(2, 18, 16)
    global_text = torch.randn(2, 16)
    word_frame = torch.randn(2, 18, 5)
    prefix_latent = torch.randn(2, 2, 32)
    prefix_mask = torch.tensor([[True, True], [True, False]])
    out = model(
        x_t,
        timesteps,
        latent_mask,
        audio=audio,
        lexical_frame=lexical,
        global_text=global_text,
        word_frame=word_frame,
        prefix_latent=prefix_latent,
        prefix_mask=prefix_mask,
    )
    assert out.shape == x_t.shape
    assert torch.isfinite(out).all()


def test_latent_flow_force_drop_paths_are_finite() -> None:
    model = LatentRectifiedFlowTransformer(
        latent_dim=16,
        audio_dim=12,
        lexical_dim=8,
        global_text_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_double_layers=1,
        token_refiner_layers=1,
        num_heads=4,
        dropout=0.0,
        local_attn_window=2,
    )
    model.eval()
    out = model(
        torch.randn(1, 4, 16),
        torch.rand(1),
        torch.ones(1, 4, dtype=torch.bool),
        audio=torch.randn(1, 7, 12),
        lexical_frame=torch.randn(1, 7, 8),
        global_text=torch.randn(1, 8),
        word_frame=torch.randn(1, 7, 5),
        force_drop_all=True,
    )
    assert out.shape == (1, 4, 16)
    assert torch.isfinite(out).all()
