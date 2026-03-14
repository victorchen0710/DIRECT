import argparse, json, re
from pathlib import Path
from deepmultilingualpunctuation import PunctuationModel

EOS = {".", "?", "!"}

def load_words_list(p: Path):
    arr = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    words = []
    for w in arr:
        if "start" not in w or "end" not in w:
            continue
        tok = str(w.get("word", "")).strip()
        if not tok:
            continue
        words.append({
            "word": tok,
            "start": float(w["start"]),
            "end": float(w["end"]),
            "score": float(w.get("score", 1.0))
        })
    words.sort(key=lambda x: x["start"])
    return words

def tokenize_with_punct(text: str):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]", text)

def is_decimal_dot(prev_tok, next_tok):
    return prev_tok.isdigit() and next_tok.isdigit()

def build_word_index_map(words):
    def norm(w):
        return re.sub(r"^[^\w']+|[^\w']+$", "", w.strip()).lower()
    return [norm(w["word"]) for w in words], norm

def norm(w: str) -> str:
    w = w.strip().lower()
    # 只保留字母数字和 '（先保留）
    w = re.sub(r"^[^\w']+|[^\w']+$", "", w)

    # 处理常见所有格/缩写：phone's -> phone, it's -> it, I'm -> im
    w = re.sub(r"'s$", "", w)      # 所有格
    w = re.sub(r"n't$", "nt", w)   # don't -> dont（可选）
    w = w.replace("'", "")         # 去掉剩余的引号

    # 复数简单归一：phones -> phone（很粗暴但有效）
    if len(w) > 3 and w.endswith("s"):
        w = w[:-1]

    return w



def split_words_by_punct_model(words,
                               model: PunctuationModel,
                               pad=0.5,
                               min_words=5,
                               max_words=30,
                               max_dur=10.0,
                               dur_total=0,
                               hard_gap=0.9,
                               ):
    if not words:
        return []

    raw_text = " ".join([w["word"] for w in words]).strip()
    punct_text = model.restore_punctuation(raw_text)
    toks = tokenize_with_punct(punct_text)

    src_norm, norm_fn = build_word_index_map(words)

    wi = 0
    w0 = 0
    segs = []

    def flush(w1):
        nonlocal w0
        if w1 <= w0:
            return
        if (w1 - w0) < min_words:
            return

        a = w0
        b = w1

        def push_seg(sa, sb):
            """把 words[sa:sb] 推入 segs；若本段很短，尝试并入上一段"""
            if sb <= sa:
                return
            if (sb - sa) < min_words:
                return

            st = max(0.0, words[sa]["start"] - pad)
            ed = words[sb - 1]["end"] + pad
            if (ed - st) > max_dur:
                return

            # 本段文本
            text = " ".join(words[i]["word"] for i in range(sa, sb)).strip()

            # ---- 合并策略：如果本段很短，尝试并入上一段 ----
            dur = ed - st
            # is_short = (sb -sa+dur) < max_dur  # 你可以调阈值

            if len(segs) > 0:
                prev = segs[-1]
                m0 = prev["w0"]
                m1 = sb
                mst = max(0.0, words[m0]["start"] - pad)
                med = words[m1 - 1]["end"] + pad
                if (med - mst) + dur <= max_dur:
                    prev["w1"] = m1
                    prev["start"] = mst
                    prev["end"] = min(med, dur_total)
                    prev["text"] = " ".join(words[i]["word"] for i in range(m0, m1)).strip()
                    return

            # 不合并就新建
            segs.append({"start": st, "end": min(ed, dur_total), "text": text, "w0": sa, "w1": sb})

        # 过长就按 max_words 拆
        while (b - a) > max_words:
            bb = a + max_words
            push_seg(a, bb)
            a = bb

        push_seg(a, b)
        w0 = w1


    for i, t in enumerate(toks):
        if t in EOS:
            prev_tok = toks[i - 1] if i - 1 >= 0 else ""
            next_tok = toks[i + 1] if i + 1 < len(toks) else ""
            if t == "." and is_decimal_dot(prev_tok, next_tok):
                continue
            flush(wi)
            continue

        nt = norm(t)
        if not nt:
            continue

        j = wi
        # print(src_norm)
        while j < len(src_norm) and (src_norm[j] != nt and norm(src_norm[j]) != nt):
            j += 1
        if j < len(src_norm):
            wi = j + 1

        # 兜底：长度/时长控制
        if wi > w0:
            dur = words[wi - 1]["end"] - words[w0]["start"]
            if (wi - w0) >= max_words or dur >= max_dur:
                flush(wi)

        # 兜底：大停顿（辅助）
        # if wi >= 2:
        #     gap = words[wi - 1]["start"] - words[wi - 2]["end"]
        #     if gap >= hard_gap:
        #         flush(wi)

    flush(wi if wi > w0 else len(words))
    return segs

def resolve_base(in_manifest: Path, item: dict):
    """
    更稳的 base 推断：
    - 优先用 manifest 里路径的公共前缀（beat/ 开头）
    - 否则退回到 in_manifest 的上两级
    """
    # 你的 words_json 通常长这样：beat/beat_english.../whisperx_json/...
    wj = item.get("words_json", "")
    if wj.startswith("beat/"):
        # 直接以项目根为 base：/home2/chenwq/projects/direct
        # 这里用 cwd 推断（你在 direct 目录下运行最稳）
        return Path(".").resolve()
    return in_manifest.parent.parent.resolve()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", default="manifests/train_whisperx.jsonl")
    ap.add_argument("--out_manifest", default="manifests/train_whisperx_split.jsonl")
    ap.add_argument("--model", default="oliverguhr/fullstop-punctuation-multilang-large")
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--min_words", type=int, default=1)
    ap.add_argument("--max_words", type=int, default=50)
    ap.add_argument("--max_dur", type=float, default=12.0)
    ap.add_argument("--hard_gap", type=float, default=0.5)
    args = ap.parse_args()

    model = PunctuationModel(model=args.model)

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out, n_drop = 0, 0, 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            item = json.loads(line)

            base = resolve_base(in_path, item)

            wj = item.get("words_json", None)
            if not wj:
                n_drop += 1
                continue
            wj_path = base / wj
            if not wj_path.exists():
                print("[WARN] words_json not found:", wj_path)
                n_drop += 1
                continue

            words = load_words_list(wj_path)
            if len(words) < args.min_words:
                n_drop += 1
                continue
            
            dur_total = float(item.get("duration", 0.0))
            segs = split_words_by_punct_model(
                words, model,
                pad=args.pad,
                min_words=args.min_words,
                max_words=args.max_words,
                max_dur=args.max_dur,
                hard_gap=args.hard_gap,
                dur_total=dur_total

            )
            if not segs:
                n_drop += 1
                continue

            for i, s in enumerate(segs):
                out_item = dict(item)
                out_item["seg_id"] = f'{item.get("id","")}_seg{i:04d}'
                out_item["start"] = s["start"]
                out_item["end"] = s["end"]
                out_item["text"] = s["text"]
                out_item["words_range"] = [s["w0"], s["w1"]]
                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"[DONE] in={n_in} out={n_out} dropped={n_drop}")
    print(f"[OUT] {out_path}")

if __name__ == "__main__":
    main()
