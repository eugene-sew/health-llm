"""Generate the single Colab notebook:  python make_notebooks.py   -> notebooks/health_llm.ipynb
One notebook, organised around the six Section C questions. Each section: plain words first, technical detail second, then code."""
import json, os

CELLS = []
md = lambda s: CELLS.append(("md", s))
code = lambda s: CELLS.append(("code", s))

md("""# Building a Health-Domain Language Model
**ICS554 NLP · Prosit 1 · Section C.** We take an open LLM (**Qwen3.5-2B-Base**) and teach it health, then prove it got better.

> **In plain words.** A language model is a program that guesses the next word. We take one that already reads English, make it read a lot of *health* text (medical papers, health guides, Ghana's treatment guidelines), then practise answering health questions. Finally we test it on material it has never seen, and compare it with the untouched model.

**How to run:** Runtime → *Change runtime type* → **T4 GPU**. Put `colab_health_bundle.zip` in Google Drive under `MyDrive/health-llm/`. Then run the cells top to bottom (Runtime → *Run all*). Progress is saved to Drive, so if Colab disconnects, just run all again and it resumes.

| Step | Prosit question | What happens | Cells |
|---|---|---|---|
| 0 | | Install and load the data | Setup |
| 1 | Q1 Data | Look at exactly what we train and test on | Data |
| 2 | Q2 Approaches | Why this recipe and not the others | (text) |
| 3 | Q3 Training | Stage 1: read health text. Stage 2: practise answering. Watch the loss fall | Train |
| 4 | Q4 Evaluation | Test base vs +Stage 1 vs +Stage 1+2 on unseen data | Evaluate |
| 5 | Q5 Results | Tables, charts, and a draft results paragraph | Results |
| 6 | Q6 Caveats | What you must know before trusting any of this | (text) |

Each section ends with a **Draft answer** written to the prosit's length limit. Rewrite it in your own words for the report.""")

# ---------------------------------------------------------------- setup
md("## Step 0 · Setup\nInstall the libraries, mount Drive, unpack the data, choose the model.")
code('''!pip install -q --upgrade --no-cache-dir unsloth unsloth_zoo
!pip install -q -U "transformers>=5"      # Qwen3.5 needs transformers v5; if Colab asks to restart the runtime, do it, then run from the next cell''')
code('''from google.colab import drive
drive.mount('/content/drive')
import os, sys, zipfile, glob, gc, json, collections
BASE = '/content/drive/MyDrive/health-llm'            # the zip lives here; checkpoints and results are saved here too
os.makedirs(BASE, exist_ok=True)
if not os.path.exists('cpt_train.jsonl'):
    zipfile.ZipFile(f'{BASE}/colab_health_bundle.zip').extractall('.')
sys.path.insert(0, '.')
import hl_lib

# ---- the knobs (everything you might want to change is here) ----
MODEL = 'unsloth/Qwen3.5-2B-Base'   # BASE model = pre-trained only, the right start for continued pre-training.
                                    # 4B does not fit a T4: Unsloth forces float32 on Qwen3.5 there (no bf16), ~16 GB of weights vs 15 GB VRAM.
LOAD = dict(load_in_4bit=False, load_in_16bit=True, full_finetuning=False)   # Unsloth advises against 4-bit on Qwen3.5
LORA = dict(r=32, lora_alpha=64, lora_dropout=0, use_gradient_checkpointing='unsloth', random_state=3407,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
BLOCK = 2048            # tokens per training block
TOKEN_BUDGET = 3_000_000  # Stage 1 reads this many tokens. Biggest lever on run time (fp32 on a T4 is slow): lower to finish sooner.
MAX_SFT, MAX_LEN = 3000, 1536   # Stage 2: number of Q&A examples, and max tokens per example
N_MCQ = 300             # exam questions per task in evaluation (500 available)
!nvidia-smi -L''')

# ---------------------------------------------------------------- Q1
md("""## Step 1 · Q1. What data did you use?

> **In plain words.** Three kinds of text. (1) *Reading material* for Stage 1: medical research papers, patient-friendly health guides, and Ghana's official treatment guidelines. (2) *Practice questions and answers* for Stage 2. (3) *Exam papers we hide from the model* to test it. The hidden material is set aside **before** any training, so the model can never memorise the test.

**Technical.**
- **Stage 1 corpus (continued pre-training):** PubMed Central open-access articles (per-article licence kept; CC BY, CC0, CC BY-NC, CC BY-NC-SA, no ND), MedlinePlus health topics (public domain, NLM), Ghana Standard Treatment Guidelines and Essential Medicines List 2017 (repeated x3 because it is small but is our local-context source).
- **Stage 2 corpus (instruction tuning):** 9,000 medical Q&A examples from medical-o1-reasoning-SFT, MedText, MedQuAD, and the *train* splits of MedMCQA and MedQA. Each is rendered in one fixed ChatML format with a GP-style system prompt.
- **Held out, never trained on:** documents from every Stage 1 source, MedQA test, MedMCQA validation, PubMedQA labeled, plus 22 hand-written GP vignettes. WikiText-2 (general English) is the **forgetting control**.
- **Deliberately excluded:** scraped patient-doctor chats (unclear licence and privacy), StatPearls (NC-ND), copyrighted textbooks.""")
code('''cpt, held = hl_lib.load_jsonl('cpt_train.jsonl'), hl_lib.load_jsonl('ppl_heldout.jsonl')
sft, sft_val, mcq_all, vign = (hl_lib.load_jsonl(f) for f in ('sft_train.jsonl', 'sft_val.jsonl', 'mcq_eval.jsonl', 'vignettes.jsonl'))
words = collections.Counter()
for d in cpt: words[d['source']] += len(d['text'].split())
print('STAGE 1 (read health text)            docs   words')
for s, n in collections.Counter(d['source'] for d in cpt).items(): print(f'  {s:<32}{n:>5}  {words[s]:>9,}')
print(f'STAGE 2 (practise Q&A)                {len(sft):,} train + {len(sft_val)} validation examples')
print('HELD OUT (never trained on)')
for s, n in collections.Counter(d['source'] for d in held).items(): print(f'  perplexity text, {s:<19}{n:>4} docs')
for t, n in collections.Counter(x['task'] for x in mcq_all).items(): print(f'  exam questions, {t:<20}{n:>4}')
print(f'  GP vignettes                        {len(vign):>4}')
print('\\nOne Stage 2 example:\\n' + hl_lib.chatml(sft[0]['messages'])[:700])''')
md("""**Draft answer (Q1, 1 paragraph).** We adapted Qwen3.5-2B-Base using open health text in two stages. Stage 1 used about 14 million words: PubMed Central open-access articles, MedlinePlus health topics, and the Ghana Standard Treatment Guidelines and Essential Medicines List (upweighted x3 for local context). Stage 2 used 9,000 medical question-and-answer examples in one ChatML format from medical-o1-reasoning-SFT, MedText, MedQuAD, and the training splits of MedMCQA and MedQA. Held-out material was split off before training: documents from each Stage 1 source, the MedQA test set, the MedMCQA validation set, PubMedQA, 22 GP vignettes, and WikiText-2 as a general-English control. We checked every licence and excluded scraped chat data and non-commercial-no-derivatives sources.""")

# ---------------------------------------------------------------- Q2
md("""## Step 2 · Q2. What approaches could you have taken, and which did you choose?

> **In plain words.** There are several ways to make an AI good at health. Cheapest: just *ask it nicely* (prompt it). Fancier: let it *look things up* in a health library while answering. Heaviest: *build a new model from zero*, or *re-teach every one of its billions of settings*. We picked a middle road: keep the model, add a small "health layer" on top (LoRA), teach that layer by having the model read health text, then practise Q&A.

| Approach | Idea | Why we did / did not use it |
|---|---|---|
| Prompt only (zero/few-shot) | No training; give instructions in the prompt | Cheap, but nothing is *learned*, so it cannot be shown to be a better health **language model**. Kept as our baseline (Model A). |
| Retrieval (RAG) | Look up documents at answer time | Useful for facts, but the prosit asks for a domain-adapted *model*; it also adds a retrieval system to build and evaluate. |
| Train from scratch | Random weights, our data | Impossible: 14M words is tiny; real LLMs need trillions of tokens and huge compute. |
| Full fine-tuning | Update all weights | Does not fit a free 15 GB T4, and risks *catastrophic forgetting* (losing general English). |
| **Continued pre-training + LoRA, then instruction tuning (chosen)** | Same next-word objective on health text, only small low-rank matrices are trained; then Q&A practice | Fits the T4, keeps the base intact, directly improves the language model (measurable as perplexity), and the second stage makes it usable as an assistant. |

**Technical.** LoRA freezes the weight matrix `W` and learns a low-rank update `W + (alpha/r)·B·A` with `r=32` on all attention and MLP projections, so only a small fraction of parameters train. Stage 1 uses the pre-training loss on raw text (causal LM, loss on every token). Stage 2 uses the same loss but **only on answer tokens** (prompt labels set to -100), so the model learns to answer rather than to imitate the question. We start from the **Base** (not Instruct) model so that the instruction-following comes from *our* Stage 2. Four-bit quantisation was rejected because Unsloth reports larger quantisation error for Qwen3.5. **Draft answer (Q2, 2 to 3 paragraphs).** See the bullets above; state the alternatives, then the choice and the three reasons: compute fit, measurability as a language model, and protection against forgetting.""")

# ---------------------------------------------------------------- Q3
md("""## Step 3 · Q3. How did you train, and what convinced you it was learning?

> **In plain words.** *Stage 1 is reading, Stage 2 is practice.* We show the model chunks of health text with the next word hidden and it guesses. Its "surprise" at the right word is called **loss**, and it should **go down**. We keep some chunks it never trains on (**validation**); if its surprise drops on those too, it is learning real health language, not memorising.

**Technical.** Loss is cross-entropy: `L = -(1/N) Σ log P(x_i | x_<i)`. Setup: LoRA `r=32`, AdamW-8bit, learning rate 1e-4 with cosine decay and 10 warm-up steps, effective batch 16 blocks (batch 1 x 16 gradient-accumulation steps), gradient checkpointing, one epoch, seed 3407. Documents are packed into fixed 2,048-token blocks separated by end-of-text tokens. The last 40 blocks are a validation set (same distribution, never trained on). **Evidence of learning we look for:** (1) validation loss *before* training vs after; (2) validation loss falls together with training loss (no growing gap means no memorisation); (3) later, held-out perplexity falls on unseen text while WikiText-2 does not blow up.

### 3a · Load the model and attach the LoRA layer""")
code('''import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import Dataset

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=BLOCK, **LOAD)
model = FastLanguageModel.get_peft_model(model, **LORA)
n_tr, n_all = (sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))
print(f'trainable parameters: {n_tr/1e6:.1f}M of {n_all/1e6:.0f}M ({100*n_tr/n_all:.2f}%)')''')
md("### 3b · Stage 1: continued pre-training (read health text)\nTurn documents into fixed-size blocks, hold 40 out as validation, and measure the loss **before** training (that number is what training must beat).")
code('''blocks = hl_lib.pack_blocks((tok(d['text'], add_special_tokens=False)['input_ids'] for d in cpt), BLOCK, tok.eos_token_id, budget=TOKEN_BUDGET)
train_b, val_b = blocks[:-40], blocks[-40:]
print(len(train_b), 'train blocks =', round(len(train_b) * BLOCK / 1e6, 2), 'M tokens;', len(val_b), 'validation blocks')
mk = lambda b: Dataset.from_dict({'input_ids': b, 'labels': b})
common = dict(per_device_train_batch_size=1, gradient_accumulation_steps=16, per_device_eval_batch_size=1, learning_rate=1e-4, lr_scheduler_type='cosine',
    warmup_steps=10, num_train_epochs=1, optim='adamw_8bit', weight_decay=0.01, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
    logging_steps=5, eval_strategy='steps', eval_steps=25, save_steps=25, save_total_limit=2, report_to='none', seed=3407)
trainer = Trainer(model=model, args=TrainingArguments(output_dir=f'{BASE}/cpt_ckpt', **common), train_dataset=mk(train_b), eval_dataset=mk(val_b),
                  data_collator=default_data_collator)
resume = bool(glob.glob(f'{BASE}/cpt_ckpt/checkpoint-*'))
if not resume: print('validation loss BEFORE training:', round(trainer.evaluate()['eval_loss'], 4))
trainer.train(resume_from_checkpoint=resume)
model.save_pretrained(f'{BASE}/adapter_cpt'); tok.save_pretrained(f'{BASE}/adapter_cpt')
json.dump(trainer.state.log_history, open(f'{BASE}/cpt_log.json', 'w'))
print('saved Stage 1 adapter')''')
code('''import matplotlib.pyplot as plt
def plot_loss(path, title, ylabel, out):
    log = json.load(open(path))
    tr = [(l['step'], l['loss']) for l in log if 'loss' in l]; ev = [(l['step'], l['eval_loss']) for l in log if 'eval_loss' in l]
    plt.plot(*zip(*tr), label='train loss'); plt.plot(*zip(*ev), 'o-', label='validation loss'); plt.xlabel('optimizer step'); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.grid(alpha=.3); plt.savefig(out, dpi=150); plt.show()
    print(f'validation loss: first {ev[0][1]:.3f} -> last {ev[-1][1]:.3f}')
plot_loss(f'{BASE}/cpt_log.json', 'Stage 1: continued pre-training on health text', 'loss (nats/token)', f'{BASE}/cpt_loss.png')''')
md("""**How to read the chart.** Both lines should slope down. If validation stops falling while train keeps falling, the model is memorising. A small, steady gap is normal.

### 3c · Stage 2: instruction tuning (practise answering)
Same model, same LoRA layer, new data: 3,000 Q&A examples rendered as `system / user / assistant`. Only the **answer** tokens count toward the loss.""")
code('''MAXLEN = MAX_LEN
mk2 = lambda rows: Dataset.from_list([hl_lib.sft_tokenize(tok, r['messages'], MAXLEN) for r in rows])
train_ds, val_ds = mk2(sft[:MAX_SFT]), mk2(sft_val[:100])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
FastLanguageModel.for_training(model)
trainer = Trainer(model=model, args=TrainingArguments(output_dir=f'{BASE}/sft_ckpt', remove_unused_columns=False, **common), train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=lambda b: hl_lib.pad_collate(b, pad))
resume = bool(glob.glob(f'{BASE}/sft_ckpt/checkpoint-*'))
if not resume: print('validation loss BEFORE Stage 2:', round(trainer.evaluate()['eval_loss'], 4))
trainer.train(resume_from_checkpoint=resume)
model.save_pretrained(f'{BASE}/adapter_sft'); tok.save_pretrained(f'{BASE}/adapter_sft')
json.dump(trainer.state.log_history, open(f'{BASE}/sft_log.json', 'w'))
plot_loss(f'{BASE}/sft_log.json', 'Stage 2: instruction tuning', 'loss (answer tokens)', f'{BASE}/sft_loss.png')
del trainer, model; gc.collect(); torch.cuda.empty_cache()     # free the GPU for evaluation''')
md("""**Draft answer (Q3, 2 to 3 paragraphs).** State the two stages and the settings (LoRA r=32 on all attention and MLP projections, AdamW-8bit, lr 1e-4 cosine, effective batch 16 x 2,048 tokens, one epoch, T4 GPU, float32 because the T4 has no bf16). Then give the evidence: the validation loss before vs after Stage 1 (numbers printed above), training and validation curves falling together, answer-only loss falling in Stage 2, and, in Step 5, held-out perplexity falling on unseen health text while WikiText-2 stays roughly flat.""")

# ---------------------------------------------------------------- Q4
md("""## Step 4 · Q4. How did you evaluate?

> **In plain words.** We ask three versions of the model the same questions and compare: **A** the original, **B** after reading health text, **C** after reading and practising. Three tests: (1) *How surprised is each model by health text it never saw?* (lower is better). (2) *Multiple-choice medical exam questions* (higher is better). (3) *Made-up patients* whose answers we read ourselves. We also test on ordinary Wikipedia text to check the model did not forget general English.

**Technical.**
- **Perplexity** = `exp(mean negative log-likelihood per token)` on held-out PMC, MedlinePlus, Ghana STG/EML, and WikiText-2 (control). This is the standard intrinsic metric for a language model. A domain gain shows as lower perplexity on the health sets; forgetting would show as higher perplexity on WikiText-2.
- **Zero-shot MCQ accuracy** on MedQA, MedMCQA, PubMedQA: for each question we compare the next-token logits of the option letters after `Answer:` (no sampling, deterministic). Every model is scored with two prompt styles (plain and chat) and we report the better one, so no model is penalised for the wrong format. Chance: 25% (MedQA, MedMCQA), about 33% (PubMedQA; always answering 'yes' gives about 55%).
- **GP vignettes:** 12 hand-written consultations, greedy decoding, read side by side. Same held-out data, prompts, and settings for A, B and C.""")
code('''from unsloth import FastLanguageModel
sources = sorted({d['source'] for d in held})
mcq = [it for t in ('MedQA', 'MedMCQA', 'PubMedQA') for it in [x for x in mcq_all if x['task'] == t][:N_MCQ]]
results = json.load(open(f'{BASE}/eval_results.json')) if os.path.exists(f'{BASE}/eval_results.json') else {}
for name, path in [('A base', MODEL), ('B +stage1', f'{BASE}/adapter_cpt'), ('C +stage1+2', f'{BASE}/adapter_sft')]:
    if name in results: continue                # resume after a disconnect
    m, t = FastLanguageModel.from_pretrained(path, max_seq_length=3072, **LOAD); FastLanguageModel.for_inference(m)
    r = {'ppl': {s: hl_lib.perplexity(m, t, [d['text'] for d in held if d['source'] == s]) for s in sources}}
    r['mcq_plain'] = hl_lib.mcq_accuracy(m, t, mcq, 'plain'); r['mcq_chat'] = hl_lib.mcq_accuracy(m, t, mcq, 'chat')
    r['gen'] = [hl_lib.generate(m, t, v['prompt'], mode='chat' if name.startswith('C') else 'plain') for v in vign[:12]]
    results[name] = r; json.dump(results, open(f'{BASE}/eval_results.json', 'w'), indent=1)
    print(name, 'done'); del m, t; gc.collect(); torch.cuda.empty_cache()''')
md("""**Draft answer (Q4, 1 to 2 paragraphs).** We compared three models on identical held-out data: the untouched base (A), base plus Stage 1 (B), and base plus both stages (C). Intrinsic evaluation was perplexity on held-out PubMed Central, MedlinePlus and Ghana STG/EML text, with WikiText-2 as a forgetting control. Extrinsic evaluation was zero-shot multiple-choice accuracy on MedQA, MedMCQA and PubMedQA (option-letter logit scoring, best of two prompt formats), plus a hand-read comparison of answers to 12 GP vignettes. Training and test splits were separated before training, and MCQ evaluation uses test/validation splits of datasets whose train splits were used in Stage 2.""")

# ---------------------------------------------------------------- Q5
md("""## Step 5 · Q5. What results did you get?
Run the cells. Everything below is computed from your run; nothing is hard-coded.""")
code('''import numpy as np
names = list(results); tasks = ['MedQA', 'MedMCQA', 'PubMedQA']; w = 0.25
best = lambda n, t: max(results[n]['mcq_plain'][t], results[n]['mcq_chat'][t])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for i, n in enumerate(names): ax[0].bar(np.arange(len(sources)) + i * w, [results[n]['ppl'][s] for s in sources], w, label=n)
ax[0].set_xticks(np.arange(len(sources)) + w); ax[0].set_xticklabels(sources); ax[0].set_ylabel('perplexity (lower is better)'); ax[0].set_title('Held-out perplexity'); ax[0].legend()
for i, n in enumerate(names): ax[1].bar(np.arange(3) + i * w, [best(n, t) for t in tasks], w, label=n)
ax[1].set_xticks(np.arange(3) + w); ax[1].set_xticklabels(tasks); ax[1].set_ylabel('accuracy'); ax[1].set_title('Zero-shot MCQ accuracy'); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{BASE}/eval_summary.png', dpi=150); plt.show()
print('| model | ' + ' | '.join(f'PPL {s}' for s in sources) + ' | ' + ' | '.join(tasks) + ' |')
for n in names: print(f"| {n} | " + ' | '.join(f"{results[n]['ppl'][s]:.2f}" for s in sources) + ' | ' + ' | '.join(f"{best(n, t):.3f}" for t in tasks) + ' |')
A, B, C = names[0], names[1], names[2]
print('\\n--- numbers for your Q5 paragraph ---')
for s in sources: print(f"perplexity {s}: {results[A]['ppl'][s]:.2f} (A) -> {results[B]['ppl'][s]:.2f} (B), change {100*(results[B]['ppl'][s]/results[A]['ppl'][s]-1):+.1f}%")
for t in tasks: print(f"{t}: {best(A, t):.3f} (A) -> {best(B, t):.3f} (B) -> {best(C, t):.3f} (C)")''')
md("""### GP vignettes side by side
**Read these yourself.** Do the answers name plausible conditions, red-flag symptoms, and sensible next steps? Note where the base model rambles and where the tuned one is structured.""")
code('''for i, v in enumerate(vign[:12]):
    print('=' * 100); print('PATIENT:', v['prompt'])
    for n in names: print(f'--- {n}:', results[n]['gen'][i][:600].replace('\\n', ' '))''')
md("""**How to write the answer (Q5, 1 to 2 paragraphs).** Use the printed numbers. Say whether perplexity fell on the three health sets, how much, and what happened on WikiText-2 (roughly flat means no forgetting). Report MCQ accuracy for A, B, C against chance, and say honestly if gains are small: the base model has already seen a lot of biomedical text, so a modest gain is a legitimate finding. Add one sentence from the vignettes (for example, structure and red-flag advice appear after Stage 2). A 2B model will not match large models on MedQA; do not claim it does.""")

# ---------------------------------------------------------------- Q6
md("""## Step 6 · Q6. What else should we know?

> **In plain words.** This is a **school exercise, not a doctor**. It can be confidently wrong. Never use it for real medical decisions.

- **Not a clinical tool.** No clinician validated the outputs. Exam accuracy measures exam knowledge, not consultation quality; vignettes are read by hand and are subjective.
- **Hardware shaped the model.** We first tried Qwen3.5-4B-Base; on a T4 (no bf16) Unsloth forces float32, about 16 GB of weights on a 15 GB GPU, so it ran out of memory. We switched to 2B. A bigger GPU could run 4B or 9B.
- **Small, single run.** One seed, one epoch, a 3M-token Stage 1 budget and 3,000 Stage 2 examples, so differences of a point or two are inside run-to-run noise.
- **Data.** Ghana STG/EML reuse terms need checking before redistribution; PMC articles keep per-article licences (some are non-commercial); the raw data is not in the public repository. English only.
- **Leakage controls.** Held-out sets were split before training; MedQA/MedMCQA *train* splits were used for Stage 2 but only their *test/validation* splits for evaluation. Web-scale pre-training of the base model may still have seen public benchmark text, which we cannot rule out.
- **Bias and safety.** PubMed skews to research-country populations; nothing here measures fairness across groups, and the model has no safety alignment beyond the system prompt.
- **Reproduce:** `collect.py` → `prep.py` → this notebook. Code: the GitHub repository linked on page 1 of the report.

**Draft answer (Q6, 1 to 3 paragraphs).** Pick the three points that matter most to your team; the hardware story, the small single run, and the not-a-clinical-tool caveat are the strongest.""")


def nb(cells):
    def cell(t, s):
        c = {"cell_type": "markdown" if t == "md" else "code", "metadata": {}, "source": s.splitlines(True)}
        if t == "code":
            c.update(execution_count=None, outputs=[])
        return c
    return {"cells": [cell(t, s) for t, s in cells], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    json.dump(nb(CELLS), open("notebooks/health_llm.ipynb", "w"), indent=1)
    print("wrote notebooks/health_llm.ipynb")
