# backend/app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import os
import traceback
# import gzip 

from services import MotionGenerator

import utils
print("[DEBUG] using utils from:", utils.__file__, flush=True)

from utils import motion_to_bvh_string

app = FastAPI(title="DirectMotion API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,   # 你现在 fetch 没带 cookie，不需要 credentials；需要的话再改 True
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = None

@app.on_event("startup")
async def load_models():
    global generator
    # 请确认这里的路径是否正确
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GPT_CKPT = os.path.join(BASE_DIR, "checkpoints/motion_gpt_text.pt")
    VQVAE_CKPT = os.path.join(BASE_DIR, "checkpoints/vqvae_big_fk_6d_tuned.pt")
    generator = MotionGenerator(GPT_CKPT, VQVAE_CKPT)
    print("[API] Models loaded successfully!")

@app.post("/generate_motion")
async def generate_motion(
    text: str = Form(""),
    audio: UploadFile = File(...),
    start_time: float = Form(0.0),
    end_time: float = Form(0.0),
    temperature: float = Form(0.6)
):
    try:
        if generator is None:
            raise HTTPException(status_code=503, detail="Models not loaded (generator is None).")

        print(f"[API] Request: Prompt='{text}'")

        audio_bytes = await audio.read()

        motion_data = generator.generate_edited_motion(
            prompt=text,
            audio_bytes=audio_bytes,
            start_time=start_time,
            end_time=end_time,
            temperature=temperature
        )

        bvh_str = motion_to_bvh_string(motion_data)

        # 建议：先返回纯文本（稳定）
        return Response(content=bvh_str, media_type="text/plain; charset=utf-8")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generate_motion failed: {repr(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)