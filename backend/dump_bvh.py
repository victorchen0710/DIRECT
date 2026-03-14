import argparse
import os
import sys

from services import MotionGenerator
from utils import motion_to_bvh_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="path to wav")
    parser.add_argument("--text", default="A person is performing a motion")
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--end_time", type=float, default=0.0)
    parser.add_argument("--out", required=True, help="output bvh path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gpt_ckpt = os.path.join(base_dir, "checkpoints/motion_gpt_text.pt")
    vqvae_ckpt = os.path.join(base_dir, "checkpoints/vqvae_big_fk_6d_tuned.pt")

    gen = MotionGenerator(gpt_ckpt, vqvae_ckpt, device=args.device)

    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    motion = gen.generate_edited_motion(
        prompt=args.text,
        audio_bytes=audio_bytes,
        start_time=args.start_time,
        end_time=args.end_time,
        temperature=0.6,
    )
    bvh_str = motion_to_bvh_string(motion)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(bvh_str)
    print(f"[dump_bvh] saved to {args.out}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
