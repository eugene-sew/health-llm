"""Helpers shared by the Colab notebooks (health-domain LLM fine-tuning). torch is imported lazily so the pure parts run anywhere.
Self-test: python hl_lib.py"""
import json, math

SYSTEM = ("You are a helpful medical assistant acting like a general practitioner (GP). Given a patient's symptoms or a health question, "
          "give the most likely explanations, key questions to ask, sensible next steps, and warning signs that need urgent care. "
          "You provide general health information, not a substitute for a clinician.")

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def chatml(messages, add_generation_prompt=False):
    """Fixed ChatML rendering, so base, continued-pretrained and instruction-tuned models all see the identical format."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n" if add_generation_prompt else s

def pack_blocks(token_lists, block, eos_id, budget=None):
    """Concatenate documents (eos between them) and cut into fixed-length blocks; stop after `budget` tokens."""
    buf, out = [], []
    for ids in token_lists:
        buf += ids + [eos_id]
        while len(buf) >= block:
            out.append(buf[:block]); buf = buf[block:]
            if budget and len(out) * block >= budget:
                return out
    return out

def sft_tokenize(tok, messages, max_len):
    """Loss only on the assistant answer: prompt tokens get label -100."""
    p = tok(chatml(messages[:-1], add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    a = tok(messages[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return {"input_ids": (p + a)[:max_len], "labels": ([-100] * len(p) + a)[:max_len]}

def pad_collate(batch, pad_id):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id); lab = torch.full((len(batch), n), -100); att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"]); ids[i, :k] = torch.tensor(b["input_ids"]); lab[i, :k] = torch.tensor(b["labels"]); att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

def perplexity(model, tok, texts, max_len=1024):
    """exp(mean negative log-likelihood per token) over all texts; lower is better."""
    import torch
    nll, count = 0.0, 0
    model.eval()
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            loss = model(input_ids=ids, labels=ids).loss.float().item()
        nll += loss * (ids.shape[1] - 1); count += ids.shape[1] - 1
    return math.exp(nll / count)

def mcq_prompt_text(item, mode):
    if mode == "chat":
        return chatml([{"role": "system", "content": SYSTEM}, {"role": "user", "content": item["prompt"]}], add_generation_prompt=True) + "Answer:"
    return f"{item['prompt']}\nAnswer:"

def mcq_accuracy(model, tok, items, mode="plain"):
    """Zero-shot: score each answer letter by next-token logit after 'Answer:'; returns {task: accuracy}."""
    import torch
    model.eval(); hit, tot = {}, {}
    for it in items:
        letter_ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in it["letters"]]
        ids = tok(mcq_prompt_text(it, mode), return_tensors="pt", add_special_tokens=False).input_ids[:, -3000:].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1]
        pred = it["letters"][int(torch.argmax(logits[letter_ids]))]
        hit[it["task"]] = hit.get(it["task"], 0) + (pred == it["answer"]); tot[it["task"]] = tot.get(it["task"], 0) + 1
    return {t: hit[t] / tot[t] for t in tot}

def generate(model, tok, prompt, mode="chat", max_new_tokens=300):
    import torch
    text = (chatml([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], add_generation_prompt=True)
            if mode == "chat" else f"Question: {prompt}\nAnswer:")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).split("<|im_end|>")[0].strip()

if __name__ == "__main__":
    assert chatml([{"role": "user", "content": "hi"}], True) == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert pack_blocks([[1, 2, 3], [4, 5, 6, 7]], 4, 0) == [[1, 2, 3, 0], [4, 5, 6, 7]] and len(pack_blocks([[1] * 50], 4, 0, budget=8)) == 2
    fake = lambda s, add_special_tokens=False: {"input_ids": list(range(len(s.split())))}
    r = sft_tokenize(fake, [{"role": "user", "content": "a b"}, {"role": "assistant", "content": "c d e"}], 99)
    assert r["labels"].count(-100) == len(chatml([{"role": "user", "content": "a b"}], True).split()) and len(r["input_ids"]) == len(r["labels"])
    it = {"prompt": "Q?", "task": "T", "letters": ["A", "B"], "answer": "A"}
    assert mcq_prompt_text(it, "plain") == "Q?\nAnswer:" and mcq_prompt_text(it, "chat").endswith("assistant\nAnswer:")
    print("hl_lib selftest ok")
