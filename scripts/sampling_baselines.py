#!/usr/bin/env python3
"""Sampling-based UQ baselines: Semantic Entropy and P(True).

These are the flagship *training-free* UQ methods in the hallucination
literature. Unlike our single-pass estimators (perplexity/entropy/MCP/CCP,
neural probes, SpatialMind), they require K stochastic decodes per sample, which
is exactly the efficiency gap we want to expose: SpatialMind matches/beats them
at 1/K the inference cost.

For each test sample (aligned to the main pipeline by dataset row order = the
sample_id used everywhere else, since get_dataset uses a fixed GLOBAL_SEED):

  Semantic Entropy (Kuhn et al. 2023):
    * sample K completions at temperature T
    * parse each into an answer, cluster by *meaning* (for these mostly
      closed-label spatial tasks, normalized answer string == meaning class;
      exact-match datasets already define the equivalence relation)
    * confidence = 1 - normalized cluster entropy (higher = more reliable)

  P(True) (Kadavath et al. 2022):
    * take the majority / first sampled answer, ask the SAME model
      "Is the proposed answer correct? (True/False)" and read the probability
      mass on the "True" token as the confidence.

Output matches baselines/combined_evaluation.json so it drops straight into
benchmark_v11.py and the LaTeX tables. Honest protocol: Platt calibration is fit
on the validation split and applied to test, identical to the other baselines.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_dataset  # noqa: E402
from scripts.metrics import compute_all_metrics  # noqa: E402
from models.calibration import StandardCalibrator  # noqa: E402


def _norm_answer(ds, text):
    try:
        a = ds.parse_answer(text)
    except Exception:
        a = (text or "").strip().lower()
    return str(a).strip().lower()


def semantic_entropy_conf(answers):
    """1 - normalized Shannon entropy over meaning clusters (closed-label => the
    normalized answer string is the cluster id). Range [0,1], higher=more sure."""
    answers = [a for a in answers if a != ""]
    if not answers:
        return 0.5
    counts = Counter(answers)
    n = sum(counts.values())
    probs = np.array([c / n for c in counts.values()], float)
    H = -np.sum(probs * np.log(probs + 1e-12))
    Hmax = math.log(len(answers))          # max entropy = all distinct
    if Hmax <= 1e-9:
        return 1.0
    return float(1.0 - H / Hmax)


def selfcheck_conf(embedder, main_text, sample_texts):
    """SelfCheckGPT (embedding-similarity variant, Manakul et al. 2023).

    Score the main response's consistency with K-1 independently sampled
    responses: high average similarity => the claim is stable across stochastic
    decodes => reliable. Uses cached samples only (no extra LLM calls) and a
    local sentence embedder. Returns a confidence in [0,1] (higher = more sure).
    """
    others = [t for t in sample_texts if t and t.strip()]
    if not main_text or not main_text.strip() or len(others) < 1:
        return 0.5
    import numpy as _np
    embs = embedder.encode([main_text] + others, normalize_embeddings=True,
                           show_progress_bar=False)
    m = embs[0]
    sims = embs[1:] @ m                    # cosine (already normalized)
    # SelfCheck reports an inconsistency score; confidence = mean similarity,
    # clipped to [0,1] (cosine of normalized embeddings is already in [-1,1]).
    return float(_np.clip(_np.mean(sims), 0.0, 1.0))


def build_engine(model_path, max_len, gpu_frac):
    from vllm import LLM
    return LLM(model=model_path, dtype="bfloat16", max_model_len=max_len,
               gpu_memory_utilization=gpu_frac, enforce_eager=False,
               trust_remote_code=True)


def apply_chat_template(tok, msgs):
    """Render chat messages, tolerating templates that reject the system role.

    Gemma / Mistral chat templates raise "System role not supported". Mirror the
    fallback used elsewhere in the pipeline (data/*.py, scripts/generate.py):
    fold any system message into the first user turn and retry, then fall back to
    plain concatenation if the template is unavailable entirely.
    """
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        pass
    # Merge system content into the first user message.
    sys_txt = " ".join(m["content"] for m in msgs if m.get("role") == "system")
    merged = []
    injected = False
    for m in msgs:
        if m.get("role") == "system":
            continue
        if m.get("role") == "user" and sys_txt and not injected:
            merged.append({"role": "user",
                           "content": f"{sys_txt}\n\n{m['content']}"})
            injected = True
        else:
            merged.append(m)
    try:
        return tok.apply_chat_template(merged, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return "\n\n".join(m["content"] for m in msgs) + "\n"


def run_split(ds, engine, tok, K, temp, max_new, ptrue_max_new, embedder=None):
    from vllm import SamplingParams
    n = len(ds.data)
    prompts = []
    for i in range(n):
        raw = ds.data[i]
        msgs = ds.build_chat_messages(raw)
        prompts.append(apply_chat_template(tok, msgs))
    # K stochastic decodes per prompt
    sp = SamplingParams(n=K, temperature=temp, top_p=0.95, max_tokens=max_new)
    outs = engine.generate(prompts, sp)
    se_conf, labels, first_ans, all_texts = [], [], [], []
    for i, o in enumerate(outs):
        raw = ds.data[i]
        gt = str(ds.get_ground_truth(raw)).strip().lower()
        texts = [c.text for c in o.outputs]
        all_texts.append(texts)                       # kept for post-hoc SelfCheck
        ans = [_norm_answer(ds, t) for t in texts]
        se_conf.append(semantic_entropy_conf(ans))
        # majority answer for P(True) + correctness label
        maj = Counter([a for a in ans if a]).most_common(1)
        pred = maj[0][0] if maj else ""
        first_ans.append(pred)
        labels.append(1 if pred == gt else 0)
    return (np.array(se_conf), np.array(labels, float), first_ans, prompts,
            all_texts)


def selfcheck_split(embedder, all_texts):
    """Post-hoc SelfCheckGPT over cached sampled texts (no GPU / no LLM calls)."""
    if embedder is None:
        return np.full(len(all_texts), 0.5)
    return np.array([selfcheck_conf(embedder, t[0], t[1:]) if t else 0.5
                     for t in all_texts])


def run_ptrue(ds, engine, tok, prompts, first_ans, max_new):
    """Ask the model to self-verify; confidence = P('True')."""
    from vllm import SamplingParams
    vprompts = []
    for i, pred in enumerate(first_ans):
        raw = ds.data[i]
        q = ds.get_question(raw)
        ctx = ds.get_context(raw)
        msg = [{"role": "user", "content":
                f"{ctx}\nQuestion: {q}\nProposed answer: {pred}\n"
                f"Is the proposed answer correct? Respond with only True or False."}]
        vprompts.append(apply_chat_template(tok, msg))
    sp = SamplingParams(temperature=0.0, max_tokens=max_new, logprobs=20)
    outs = engine.generate(vprompts, sp)
    conf = []
    for o in outs:
        p_true = 0.0
        got = False
        if o.outputs and o.outputs[0].logprobs:
            first_tok_lp = o.outputs[0].logprobs[0]
            for tid, lp in first_tok_lp.items():
                tokstr = (lp.decoded_token or "").strip().lower()
                if tokstr in ("true", "yes", "correct"):
                    p_true = max(p_true, math.exp(lp.logprob)); got = True
        conf.append(p_true if got else 0.5)
    return np.array(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_val", type=int, default=2000)
    ap.add_argument("--max_test", type=int, default=3000)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--ptrue_max_new", type=int, default=4)
    ap.add_argument("--gpu_frac", type=float, default=0.85)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--embedder_path",
                    default="spatialmind/models/all-MiniLM-L6-v2",
                    help="Sentence embedder for the SelfCheckGPT variant.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # IMPORTANT: build the vLLM engine BEFORE touching sentence-transformers.
    # Loading an embedder first initializes a CUDA context in the parent process,
    # which makes vLLM's forked workers crash with "Cannot re-initialize CUDA in
    # forked subprocess". vLLM claims CUDA first; the embedder runs on CPU after.
    engine = build_engine(args.model_path, args.max_len, args.gpu_frac)

    splits = {"validation": args.max_val, "test": args.max_test}
    raw = {}
    texts_by_split = {}
    for split, cap in splits.items():
        ds = get_dataset(name=args.dataset_name, dataset_path=args.dataset_path,
                         split=split, k_hop_values=None, max_samples=cap)
        se, y, first, prompts, all_texts = run_split(
            ds, engine, tok, args.K, args.temp, args.max_new, args.ptrue_max_new)
        pt = run_ptrue(ds, engine, tok, prompts, first, args.ptrue_max_new)
        raw[split] = {"se": se, "pt": pt, "y": y, "n": len(y)}
        texts_by_split[split] = all_texts
        print(f"[{split}] n={len(y)} pos_rate={y.mean():.3f} "
              f"SE_auroc={compute_all_metrics(y,se)['roc_auc']:.3f} "
              f"PT_auroc={compute_all_metrics(y,pt)['roc_auc']:.3f}")

    # SelfCheckGPT is computed AFTER all vLLM generation, so the CPU embedder's
    # CUDA-free context is created only once vLLM is done with the GPU.
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedder_path, device="cpu")
        print(f"[selfcheck] embedder loaded from {args.embedder_path}")
    except Exception as e:
        print(f"[selfcheck] embedder unavailable ({e}); SelfCheck -> 0.5")
    for split in splits:
        sc = selfcheck_split(embedder, texts_by_split[split])
        raw[split]["sc"] = sc
        print(f"[{split}] SC_auroc={compute_all_metrics(raw[split]['y'],sc)['roc_auc']:.3f}")

    # Platt-calibrate each method: fit on validation, apply to test (honest).
    os.makedirs(args.out_dir, exist_ok=True)
    combined = {}
    for method, key in [("semantic_entropy", "se"), ("selfcheckgpt", "sc"),
                        ("p_true", "pt")]:
        v, t = raw["validation"], raw["test"]
        cal = StandardCalibrator().fit(np.clip(v[key], 1e-6, 1 - 1e-6), v["y"])
        s_test = cal.transform(np.clip(t[key], 1e-6, 1 - 1e-6))
        m = compute_all_metrics(t["y"], s_test)
        combined[method] = {
            "method_type": "baseline_sampling", "head_type": method,
            "split": "test", "total_samples": int(t["n"]),
            "calibration": {"mode": "standard", "fit_on": "validation"},
            "overall_metrics": m,
            "predictions": [{"sample_id": i, "trace_label": int(t["y"][i]),
                             "trace_score": float(s_test[i])} for i in range(t["n"])],
        }
        print(f"{method:18s} test AUROC={m['roc_auc']:.3f} Brier={m['brier']:.3f}")
    json.dump(combined, open(f"{args.out_dir}/combined_evaluation.json", "w"), indent=2)
    print(f"wrote {args.out_dir}/combined_evaluation.json")


if __name__ == "__main__":
    main()
