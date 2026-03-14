import argparse
import json
import os
from pathlib import Path
import math
import torch.multiprocessing as mp
import gc
import traceback

# ================= Configuration =================
GPU_INDICES = [0, 1, 2, 3]
BATCH_SIZE = 16
COMPUTE_TYPE = "float16"

# 你的本地模型目录：里面直接有 model.bin / config.json 等
MODELS_ROOT = Path("~/projects/direct/models/whisperx").expanduser().resolve()
MODEL_DIR = MODELS_ROOT  # 这里就是模型目录本身

# 让 huggingface_hub / transformers 的缓存也写到这个目录里（对齐模型也会受益）
HF_HOME = MODELS_ROOT / "hf_home"
HF_HUB_CACHE = HF_HOME / "hub"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def process_chunk(gpu_id, files_chunk, args):
    """
    子进程：处理分配到的文件列表
    关键点：
    - 先设置 CUDA_VISIBLE_DEVICES，让每个进程只看到一张卡
    - CTranslate2 的 device 只能用 "cuda"/"cpu"，不能用 "cuda:0"
    """
    # 必须尽量早设置，让后续 import 的库看到正确 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 延迟 import：避免主进程 import 时就初始化 CUDA / 绑定设备
    import torch
    import whisperx

    device = "cuda"
    print(f"[GPU {gpu_id}] Initializing model on {device} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})...")

    try:
        # 0) 检查本地模型是否存在
        if not (MODEL_DIR / "model.bin").exists():
            raise FileNotFoundError(
                f"Local model not found: {MODEL_DIR}/model.bin does not exist. "
                f"Please ensure the faster-whisper model files are placed directly under {MODEL_DIR}"
            )

        # 1) 加载 ASR 模型：直接从本地目录加载
        # 这里传入必须是 str，不要传 Path
        model = whisperx.load_model(
            str(MODEL_DIR),
            device,  # 你现在是 "cuda"
            compute_type=COMPUTE_TYPE,
            language="en",          # 你是 BEAT English，指定语言可显著加速（避免每条音频检测语言）
            vad_method="silero",    # 关键：别用 pyannote
            download_root=str(MODELS_ROOT),
        )

        # 2) 加载对齐模型（会用 HF_HOME/HF_HUB_CACHE）
        # 注意：这里 device 也用 "cuda"，不要用 "cuda:0"
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig

        with torch.serialization.safe_globals([ListConfig, DictConfig]):
            model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        print(f"[GPU {gpu_id}] Loaded. Processing {len(files_chunk)} files...")

        processed_cnt = 0
        beat_root = Path(args.beat_root)

        # 输出目录
        if args.output_root:
            out_root = Path(args.output_root)
        else:
            out_root = beat_root / "whisperx_json"

        for wav_path in files_chunk:
            try:
                try:
                    rel_path = wav_path.relative_to(beat_root)
                except ValueError:
                    rel_path = Path(wav_path.name)

                save_path = out_root / rel_path.with_suffix(".json")
                if save_path.exists():
                    continue

                save_path.parent.mkdir(parents=True, exist_ok=True)

                audio = whisperx.load_audio(str(wav_path))

                # 1) 转录
                result = model.transcribe(audio, batch_size=BATCH_SIZE)

                # 2) 对齐
                if not result.get("segments"):
                    word_segments = []
                else:
                    result_aligned = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )
                    word_segments = result_aligned["word_segments"]

                # 3) 保存
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(word_segments, f, indent=2)

                processed_cnt += 1
                if processed_cnt % 50 == 0:
                    print(f"[GPU {gpu_id}] Progress: {processed_cnt}/{len(files_chunk)}")
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {wav_path}: {e}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Critical Error during init: {e}")
        traceback.print_exc()

    print(f"[GPU {gpu_id}] Finished Chunk.")


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--beat_root", type=str, required=True, help="BEAT数据集所在的文件夹")
    parser.add_argument("--output_root", type=str, default=None, help="可选：指定输出目录")
    args = parser.parse_args()

    beat_path = Path(args.beat_root)
    if not beat_path.exists():
        raise FileNotFoundError(f"找不到目录: {beat_path}")

    print(f"[Main] Scanning audio files recursively in {beat_path}...")
    all_wavs = list(beat_path.rglob("*.wav"))
    all_wavs = [p for p in all_wavs if not p.name.startswith("._")]

    total_files = len(all_wavs)
    if total_files == 0:
        raise FileNotFoundError(f"在 {beat_path} 下没有找到任何 .wav 文件！")

    print(f"[Main] Found {total_files} wav files.")

    num_gpus = len(GPU_INDICES)
    chunk_size = math.ceil(total_files / num_gpus)

    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        chunks.append(all_wavs[start_idx:end_idx])

    print(f"[Main] Launching {num_gpus} processes...")

    processes = []
    for i, gpu_id in enumerate(GPU_INDICES):
        if not chunks[i]:
            continue
        p = mp.Process(target=process_chunk, args=(gpu_id, chunks[i], args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[Main] All Done!")


if __name__ == "__main__":
    main()
