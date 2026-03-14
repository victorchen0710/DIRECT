import json
from pathlib import Path
from praatio import textgrid

def extract_tier(tg, name):
    tier = tg.getTier(name)
    out = []
    for s, e, lab in tier.entries:
        lab = (lab or "").strip()
        out.append({"lab": lab, "s": float(s), "e": float(e)})
    return out

def main(root_dir: str):
    root = Path(root_dir)
    for tg_path in root.rglob("*.TextGrid"):
        try:
            tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

            words = extract_tier(tg, "words")   # lab 为空也保留（你后面可决定是否转 <PAUSE>）
            phones = extract_tier(tg, "phones")

            out = {
                "words": words,    # 每个元素: {"lab": "the"/""/..., "s":..., "e":...}
                "phones": phones   # 每个元素: {"lab": "DH"/""/..., "s":..., "e":...}
            }

            out_path = tg_path.with_suffix(".align.json")
            out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except Exception as ex:
            print(f"[WARN] {tg_path}: {ex}")

if __name__ == "__main__":
    main("beat/beat_english_v0.2.1")

