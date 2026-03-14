import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import io
from transformers import Wav2Vec2Processor, Wav2Vec2Model, CLIPTokenizerFast, CLIPTextModel
from scipy.signal import savgol_filter

# 路径黑魔法：确保能导入 stageA 和 stageB
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from stageA.train_stage1_vqvae import MotionVQVAE
# 假设 MotionGPT 类定义在 stageB.train_gpt 中，如果报错请将 MotionGPT 类定义直接贴在这里
from stageB.train_gpt import MotionGPT 

class MotionGenerator:
    def __init__(self, gpt_ckpt, vqvae_ckpt, device='cuda'):
        self.device = device
        print(f"[Backend] Initializing models on {device}...")
        
        # 1. Load VQ-VAE
        vq_data = torch.load(vqvae_ckpt, map_location='cpu')
        vq_args = vq_data['args']
        self.vqvae = MotionVQVAE(
            motion_dim=vq_data['keep_dim'],
            hidden=vq_args['hidden'],
            code_dim=vq_args['code_dim'],
            n_codes=vq_args['n_codes'],
            n_downsample=vq_args['n_downsample']
        ).to(device)
        self.vqvae.load_state_dict(vq_data['model'], strict=False)
        self.vqvae.eval()
        
        self.mean = torch.from_numpy(vq_data['mean']).to(device)
        self.std = torch.from_numpy(vq_data['std']).to(device)
        self.n_downsample = vq_args['n_downsample']

        # 2. Load GPT
        self.gpt = MotionGPT(n_codes=vq_args['n_codes']).to(device)
        self.gpt.load_state_dict(torch.load(gpt_ckpt, map_location='cpu'))
        self.gpt.eval()

        # 3. Load Audio Processor (Local)
        print("[Backend] Loading Wav2Vec2...")
        self.w2v_proc = Wav2Vec2Processor.from_pretrained("stageB/models/wav2vec2")
        self.w2v_model = Wav2Vec2Model.from_pretrained("stageB/models/wav2vec2").to(device)
        self.w2v_model.eval()

        # 4. Load Text Tokenizer (Local)
        print("[Backend] Loading CLIP...")
        self.tokenizer = CLIPTokenizerFast.from_pretrained("stageB/models/clip")

    def process_audio(self, audio_bytes):
        """将上传的音频字节流转换为 768维特征"""
        speech, sr = sf.read(io.BytesIO(audio_bytes))
        
        # 重采样到 16k
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        
        # 单声道处理
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
            
        inputs = self.w2v_proc(speech, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        with torch.no_grad():
            outputs = self.w2v_model(inputs)
            feat = outputs.last_hidden_state[0].cpu().numpy()
            
        # Resample to 15 FPS
        duration = len(speech) / 16000
        src_len = feat.shape[0]
        tgt_len = int(duration * 15)
        
        x_src = np.linspace(0, duration, src_len)
        x_tgt = np.linspace(0, duration, tgt_len)
        
        feat_15fps = np.zeros((tgt_len, 768), dtype=np.float32)
        for d in range(768):
            feat_15fps[:, d] = np.interp(x_tgt, x_src, feat[:, d])
            
        return torch.from_numpy(feat_15fps).float().to(self.device).unsqueeze(0)

    @torch.no_grad()
    def _generate_core(self, text, audio_feat, temperature=0.6, rep_penalty=1.1):
        """内部核心生成函数"""
        stride = 2 ** self.n_downsample
        audio_down = audio_feat[:, ::stride, :]
        seq_len = audio_down.size(1)
        
        text_inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
        text_feat = self.gpt.clip(**text_inputs).pooler_output
        text_emb = self.gpt.text_proj(text_feat).unsqueeze(1) + self.gpt.pos_emb[:, :1, :]
        audio_emb = self.gpt.audio_proj(audio_down) + self.gpt.pos_emb[:, 1:1+seq_len, :]
        memory = torch.cat([text_emb, audio_emb], dim=1)
        
        sos = self.gpt.sos_token
        gpt_input = sos
        generated = []
        
        for t in range(seq_len):
            curr_len = gpt_input.size(1)
            gpt_input_pos = gpt_input + self.gpt.pos_emb[:, :curr_len, :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr_len).to(self.device)
            out = self.gpt.decoder(tgt=gpt_input_pos, memory=memory, tgt_mask=tgt_mask)
            logits = self.gpt.out_head(out[:, -1, :])
            
            # Repetition Penalty
            if len(generated) > 0 and rep_penalty != 1.0:
                for token_id in set(generated[-50:]):
                    if token_id < logits.size(-1):
                        if logits[0, token_id] < 0: logits[0, token_id] *= rep_penalty
                        else: logits[0, token_id] /= rep_penalty

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            
            idx = torch.tensor([[next_token]], device=self.device)
            gpt_input = torch.cat([gpt_input, self.gpt.code_emb(idx)], dim=1)

        codes = torch.tensor([generated], dtype=torch.long, device=self.device)
        z_q = self.vqvae.vq.codebook(codes).permute(0, 2, 1)
        motion = self.vqvae.dec(z_q).permute(0, 2, 1)
        motion = motion * self.std + self.mean
        return motion[0].cpu().numpy()

    def generate_edited_motion(self, prompt, audio_bytes, start_time, end_time, temperature=0.6):
        """对外接口：支持时间轴编辑"""
        # 1. 提取音频特征
        audio_feat = self.process_audio(audio_bytes)
        
        # 2. Base Layer: 音频驱动的自然动作
        print("[Backend] Generating Base Layer...")
        base_motion = self._generate_core("A person is performing a motion", audio_feat, temperature)
        
        # 3. 如果无需编辑，直接平滑返回
        if not prompt or start_time >= end_time:
            return savgol_filter(base_motion, window_length=15, polyorder=2, axis=0)

        # 4. Edit Layer: 用户指令驱动的动作
        print(f"[Backend] Generating Edit Layer for '{prompt}'...")
        edit_motion = self._generate_core(prompt, audio_feat, temperature)
        
        # 5. 缝合 (Stitching)
        FPS = 15
        start_frame = int(start_time * FPS)
        end_frame = int(end_time * FPS)
        T = len(base_motion)
        
        start_frame = max(0, min(start_frame, T))
        end_frame = max(0, min(end_frame, T))
        
        final_motion = base_motion.copy()
        
        if end_frame > start_frame:
            print(f"[Backend] Stitching frames {start_frame}-{end_frame}")
            
            # 简单的淡入淡出混合 (Cross-fade) 防止跳变
            blend_len = min(5, (end_frame - start_frame) // 2)
            
            # 中间替换
            if end_frame - blend_len > start_frame + blend_len:
                final_motion[start_frame+blend_len : end_frame-blend_len] = edit_motion[start_frame+blend_len : end_frame-blend_len]
            
            # 前端混合 (Base -> Edit)
            for i in range(blend_len):
                alpha = i / blend_len
                idx = start_frame + i
                if idx < T:
                    final_motion[idx] = base_motion[idx] * (1-alpha) + edit_motion[idx] * alpha
                    
            # 后端混合 (Edit -> Base)
            for i in range(blend_len):
                alpha = i / blend_len
                idx = end_frame - blend_len + i
                if idx < T:
                    final_motion[idx] = edit_motion[idx] * (1-alpha) + base_motion[idx] * alpha

        # 6. 全局平滑
        final_motion = savgol_filter(final_motion, window_length=15, polyorder=2, axis=0)
        return final_motion