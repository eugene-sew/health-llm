"""Build the training / held-out / evaluation sets for the Colab notebooks (run locally after collect.py).
  .venv/bin/python prep.py   |   .venv/bin/python prep.py --selftest
Outputs data/prep/: cpt_train.jsonl, ppl_heldout.jsonl, sft_train.jsonl, sft_val.jsonl, mcq_eval.jsonl, vignettes.jsonl, report.json
and colab_health_bundle.zip. Held-out sets are split BEFORE training so perplexity is measured on text the model never saw.
"""
import ast, glob, hashlib, json, os, random, sys, zipfile
import pyarrow.parquet as pq

RAW, OUT = "data/raw", "data/prep"
from hl_lib import SYSTEM          # single source of truth for the prompt
GHANA_UPWEIGHT = 3          # the Ghana guidelines are tiny (about 1% of words); repeat them so the model sees them

def h(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

def jl(path):
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except ValueError:          # partial PMC file may end mid-line if the sampler was stopped
            pass
    return out

def parquet_rows(pattern):
    for f in sorted(glob.glob(f"{RAW}/hf/{pattern}")):
        yield from pq.read_table(f).to_pylist()

def windows(text, size=400, min_words=100):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size) if len(w[i:i + size]) >= min_words]

def options_text(opts):
    return "\n".join(f"{k}. {v}" for k, v in opts.items())

def mcq_prompt(question, opts):
    return f"{question}\n{options_text(opts)}\nAnswer with the letter of the correct option."

def write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)

def build():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)
    stats = {}
    # ---- corpus split (held-out first)
    pmc_path = f"{RAW}/pmc_oa.jsonl" if os.path.exists(f"{RAW}/pmc_oa.jsonl") else f"{RAW}/pmc_oa.partial.jsonl"
    pmc = jl(pmc_path)
    cpt, held, n_pmc_held = [], [], 0
    for d in pmc:
        item = {"source": "pmc_oa", "id": d["id"], "text": d["text"]}
        if h(d["id"]) % 20 == 0 and n_pmc_held < 150:       # ~5% of articles, capped at 150
            held.append(item); n_pmc_held += 1
        else:
            cpt.append(item)
    for d in jl(f"{RAW}/medlineplus.jsonl"):
        (held if h(d["id"]) % 10 == 0 else cpt).append({"source": "medlineplus", "id": d["id"], "text": d["text"]})
    for d in jl(f"{RAW}/moh_ghana.jsonl"):
        for i, w in enumerate(windows(d["text"])):          # every 10th window is held out; the rest are repeated GHANA_UPWEIGHT times
            item = {"source": "moh_ghana", "id": f"{d['id']}-{i}", "text": w}
            if i % 10 == 0:
                held.append(item)
            else:
                cpt.extend([item] * GHANA_UPWEIGHT)
    wiki_lines = [r["text"].strip() for r in parquet_rows("wikitext2__wikitext-2-raw-v1__test__0.parquet") if r["text"].strip() and not r["text"].strip().startswith("=")]
    for i, w in enumerate(windows(" ".join(wiki_lines), min_words=200)[:150]):
        held.append({"source": "wikitext2", "id": f"wiki-{i}", "text": w})
    rng.shuffle(cpt)
    # held-out text is trimmed to about 700 words so every model is scored on the same span
    for d in held:
        d["text"] = " ".join(d["text"].split()[:700])
    stats["cpt_docs"] = write("cpt_train", cpt); stats["ppl_heldout_docs"] = write("ppl_heldout", held)
    stats["cpt_words"] = sum(len(d["text"].split()) for d in cpt)
    stats["heldout_by_source"] = {s: sum(d["source"] == s for d in held) for s in {d["source"] for d in held}}

    # ---- instruction (SFT) data
    def chat(user, assistant):
        return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}
    sft = []
    o1 = list(parquet_rows("medical_o1_sft__en__train__*.parquet")); rng.shuffle(o1)
    sft += [chat(r["Question"], r["Response"]) for r in o1[:3000]]
    mt = list(parquet_rows("medtext__default__train__*.parquet")); rng.shuffle(mt)
    sft += [chat(r["Prompt"], r["Completion"]) for r in mt[:1300]]
    val_vign = [{"prompt": r["Prompt"], "reference": r["Completion"]} for r in mt[1300:]]
    mq = [r for r in parquet_rows("medquad__default__train__*.parquet") if r["answer"] and r["question"] and 30 <= len(r["answer"].split()) <= 400]; rng.shuffle(mq)
    sft += [chat(r["question"], r["answer"]) for r in mq[:2500]]
    mm = [r for r in parquet_rows("medmcqa__default__train__*.parquet") if r["choice_type"] == "single" and r["exp"] not in (None, "None", "") and len(r["exp"].split()) >= 15]; rng.shuffle(mm)
    for r in mm[:1500]:
        opts = dict(zip("ABCD", [r["opa"], r["opb"], r["opc"], r["opd"]])); L = "ABCD"[int(r["cop"])]
        sft.append(chat(mcq_prompt(r["question"], opts), f"Answer: {L}. {opts[L]}\nExplanation: {r['exp']}"))
    mu = list(parquet_rows("medqa_usmle__default__train__*.parquet")); rng.shuffle(mu)
    for r in mu[:1000]:
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        sft.append(chat(mcq_prompt(r["question"], opts), f"Answer: {r['answer_idx']}. {opts[r['answer_idx']]}"))
    rng.shuffle(sft)
    stats["sft_train"] = write("sft_train", sft[300:]); stats["sft_val"] = write("sft_val", sft[:300])

    # ---- MCQ evaluation (never used for training): test/validation splits only
    ev = []
    for r in list(parquet_rows("medqa_usmle__default__test__*.parquet")):
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        ev.append({"task": "MedQA", "prompt": mcq_prompt(r["question"], opts), "letters": list(opts), "answer": r["answer_idx"]})
    ev_mm = []
    for r in parquet_rows("medmcqa__default__validation__*.parquet"):
        if r["choice_type"] == "single":
            opts = dict(zip("ABCD", [r["opa"], r["opb"], r["opc"], r["opd"]]))
            ev_mm.append({"task": "MedMCQA", "prompt": mcq_prompt(r["question"], opts), "letters": list("ABCD"), "answer": "ABCD"[int(r["cop"])]})
    ev_pq = []
    for r in parquet_rows("pubmedqa_labeled__pqa_labeled__train__*.parquet"):
        ctx = r["context"]; ctx = ast.literal_eval(ctx) if isinstance(ctx, str) else ctx
        opts = {"A": "yes", "B": "no", "C": "maybe"}
        ev_pq.append({"task": "PubMedQA", "prompt": f"Abstract: {' '.join(ctx['contexts'])}\n\n" + mcq_prompt(r["question"], opts),
                      "letters": list("ABC"), "answer": {"yes": "A", "no": "B", "maybe": "C"}[r["final_decision"]]})
    mcq = []
    for task in (ev, ev_mm, ev_pq):
        rng.shuffle(task); mcq += task[:500]
    stats["mcq_eval"] = write("mcq_eval", mcq)

    # ---- qualitative GP vignettes: hand-written Ghana cases + held-out MedText cases
    vign = jl("vignettes.jsonl") + [{"prompt": v["prompt"], "reference": v["reference"]} for v in val_vign[:10]]
    stats["vignettes"] = write("vignettes", vign)
    json.dump(stats, open(f"{OUT}/report.json", "w"), indent=1)
    with zipfile.ZipFile("colab_health_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(f"{OUT}/*.jsonl")) + [f"{OUT}/report.json", "hl_lib.py"]:
            if os.path.exists(f):
                z.write(f, os.path.basename(f))
    print(json.dumps(stats, indent=1), "\nbundle MB:", round(os.path.getsize("colab_health_bundle.zip") / 1e6, 1))

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert windows("a " * 950)[0].count("a") == 400 and len(windows("a " * 950)) == 3
        assert mcq_prompt("Q?", {"A": "x", "B": "y"}) == "Q?\nA. x\nB. y\nAnswer with the letter of the correct option."
        assert h("x") == h("x")
        sys.exit("selftest ok")
    build()
