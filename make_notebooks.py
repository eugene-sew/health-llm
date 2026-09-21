"""Generate the Colab notebooks:  python make_notebooks.py   -> notebooks/01_cpt.ipynb, 02_sft.ipynb, 03_eval.ipynb"""
import json, os

SETUP = '''from google.colab import drive
drive.mount('/content/drive')
import os, sys, zipfile
BASE = '/content/drive/MyDrive/health-llm'            # put colab_health_bundle.zip here
os.makedirs(BASE, exist_ok=True)
if not os.path.exists('cpt_train.jsonl'):
    zipfile.ZipFile(f'{BASE}/colab_health_bundle.zip').extractall('.')
sys.path.insert(0, '.')
MODEL = 'unsloth/Qwen3.5-2B-Base'    # base (pre-trained only) model, the right start for continued pre-training. 4B does not fit a T4: Unsloth forces float32 on Qwen3.5 there (no bf16), ~16 GB of weights vs 15 GB VRAM
LOAD = dict(load_in_4bit=False, load_in_16bit=True, full_finetuning=False)   # Unsloth advises against 4-bit on Qwen3.5
LORA = dict(r=32, lora_alpha=64, lora_dropout=0, use_gradient_checkpointing='unsloth', random_state=3407,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
!nvidia-smi -L'''

INSTALL = '''!pip install -q --upgrade --no-cache-dir unsloth unsloth_zoo
!pip install -q -U "transformers>=5"      # Qwen3.5 needs transformers v5; restart the runtime if Colab asks, then skip this cell'''

NB1 = [
 ("md", "# 01 · Continued pre-training (domain adaptation) on health text\nRuntime → **T4 GPU**. Trains a LoRA adapter on the base model with the ordinary next-token objective over PubMed Central articles, MedlinePlus and the Ghana STG/EML. "
        "The validation loss curve is your evidence that the model is learning. Checkpoints go to Drive; re-running resumes."),
 ("code", INSTALL),
 ("code", SETUP),
 ("code", '''import glob, json, torch, hl_lib
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, default_data_collator
from datasets import Dataset

BLOCK, TOKEN_BUDGET = 2048, 6_000_000      # T4: budget decides run time (fp32 on a T4 is slow). Lower it to finish sooner.
model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=BLOCK, **LOAD)
model = FastLanguageModel.get_peft_model(model, **LORA)
docs = hl_lib.load_jsonl('cpt_train.jsonl')
blocks = hl_lib.pack_blocks((tok(d['text'], add_special_tokens=False)['input_ids'] for d in docs), BLOCK, tok.eos_token_id, budget=TOKEN_BUDGET)
train_b, val_b = blocks[:-40], blocks[-40:]           # last 40 blocks: validation loss curve (unseen text, same distribution)
print(len(train_b), 'train blocks =', len(train_b) * BLOCK / 1e6, 'M tokens;', len(val_b), 'validation blocks')
mk = lambda b: Dataset.from_dict({'input_ids': b, 'labels': b})'''),
 ("code", '''args = TrainingArguments(output_dir=f'{BASE}/cpt_ckpt', per_device_train_batch_size=1, gradient_accumulation_steps=16, per_device_eval_batch_size=1,
    learning_rate=1e-4, lr_scheduler_type='cosine', warmup_steps=10, num_train_epochs=1, optim='adamw_8bit', weight_decay=0.01,
    fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5, eval_strategy='steps', eval_steps=25,
    save_steps=25, save_total_limit=2, report_to='none', seed=3407)
trainer = Trainer(model=model, args=args, train_dataset=mk(train_b), eval_dataset=mk(val_b), data_collator=default_data_collator)
trainer.train(resume_from_checkpoint=bool(glob.glob(f'{BASE}/cpt_ckpt/checkpoint-*')))
model.save_pretrained(f'{BASE}/adapter_cpt'); tok.save_pretrained(f'{BASE}/adapter_cpt')
json.dump(trainer.state.log_history, open(f'{BASE}/cpt_log.json', 'w'))
print('saved', f'{BASE}/adapter_cpt')'''),
 ("code", '''import matplotlib.pyplot as plt
log = json.load(open(f'{BASE}/cpt_log.json'))
tr = [(l['step'], l['loss']) for l in log if 'loss' in l]; ev = [(l['step'], l['eval_loss']) for l in log if 'eval_loss' in l]
plt.plot(*zip(*tr), label='train loss'); plt.plot(*zip(*ev), 'o-', label='validation loss'); plt.xlabel('optimizer step'); plt.ylabel('loss (nats/token)')
plt.title('Continued pre-training on health text'); plt.legend(); plt.grid(alpha=.3); plt.savefig(f'{BASE}/cpt_loss.png', dpi=150); plt.show()'''),
]

NB2 = [
 ("md", "# 02 · Instruction tuning (supervised fine-tuning) on medical Q&A\nStarts from the adapter trained in notebook 01 and teaches the model to answer as a GP-style assistant. Loss is computed on the answer only."),
 ("code", INSTALL),
 ("code", SETUP),
 ("code", '''import glob, json, torch, hl_lib
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments
from datasets import Dataset

MAX_SFT, MAX_LEN = 6000, 1536
model, tok = FastLanguageModel.from_pretrained(f'{BASE}/adapter_cpt', max_seq_length=2048, **LOAD)   # base + stage-1 LoRA
FastLanguageModel.for_training(model)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters:', n_train)
assert n_train > 0, 'LoRA weights are frozen after loading; see Unsloth docs on continuing training from a saved adapter'
sft = hl_lib.load_jsonl('sft_train.jsonl')[:MAX_SFT]; val = hl_lib.load_jsonl('sft_val.jsonl')[:100]
mk = lambda rows: Dataset.from_list([hl_lib.sft_tokenize(tok, r['messages'], MAX_LEN) for r in rows])
train_ds, val_ds = mk(sft), mk(val)
print(len(train_ds), 'train /', len(val_ds), 'val examples')
print(hl_lib.chatml(sft[0]['messages'])[:800])'''),
 ("code", '''pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
args = TrainingArguments(output_dir=f'{BASE}/sft_ckpt', per_device_train_batch_size=1, gradient_accumulation_steps=16, per_device_eval_batch_size=1,
    learning_rate=1e-4, lr_scheduler_type='cosine', warmup_steps=10, num_train_epochs=1, optim='adamw_8bit', weight_decay=0.01,
    fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5, eval_strategy='steps', eval_steps=25,
    save_steps=25, save_total_limit=2, report_to='none', seed=3407, remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=lambda b: hl_lib.pad_collate(b, pad))
trainer.train(resume_from_checkpoint=bool(glob.glob(f'{BASE}/sft_ckpt/checkpoint-*')))
model.save_pretrained(f'{BASE}/adapter_sft'); tok.save_pretrained(f'{BASE}/adapter_sft')
json.dump(trainer.state.log_history, open(f'{BASE}/sft_log.json', 'w'))
print('saved', f'{BASE}/adapter_sft')'''),
 ("code", '''import matplotlib.pyplot as plt
log = json.load(open(f'{BASE}/sft_log.json'))
tr = [(l['step'], l['loss']) for l in log if 'loss' in l]; ev = [(l['step'], l['eval_loss']) for l in log if 'eval_loss' in l]
plt.plot(*zip(*tr), label='train loss'); plt.plot(*zip(*ev), 'o-', label='validation loss'); plt.xlabel('optimizer step'); plt.ylabel('loss (answer tokens)')
plt.title('Instruction tuning'); plt.legend(); plt.grid(alpha=.3); plt.savefig(f'{BASE}/sft_loss.png', dpi=150); plt.show()'''),
]

NB3 = [
 ("md", "# 03 · Evaluation: base vs +continued pre-training vs +instruction tuning\nSame held-out data for all three models. (1) **Perplexity** on text none of them saw: PubMed Central articles, MedlinePlus, Ghana STG/EML, and WikiText-2 as a general-English control. "
        "(2) **Zero-shot multiple-choice accuracy** on MedQA, MedMCQA, PubMedQA. (3) Side-by-side answers to GP vignettes. Results are saved to Drive as JSON and PNG."),
 ("code", INSTALL),
 ("code", SETUP),
 ("code", '''import gc, json, torch, hl_lib
from unsloth import FastLanguageModel
N_MCQ = 300                                     # questions per task (500 available). Lower if time is short.
held = hl_lib.load_jsonl('ppl_heldout.jsonl'); mcq_all = hl_lib.load_jsonl('mcq_eval.jsonl'); vign = hl_lib.load_jsonl('vignettes.jsonl')
sources = sorted({d['source'] for d in held})
mcq = [it for t in ('MedQA', 'MedMCQA', 'PubMedQA') for it in [x for x in mcq_all if x['task'] == t][:N_MCQ]]
print({s: sum(d['source'] == s for d in held) for s in sources}, '| mcq items:', len(mcq), '| vignettes:', len(vign))'''),
 ("code", '''results = json.load(open(f'{BASE}/eval_results.json')) if os.path.exists(f'{BASE}/eval_results.json') else {}
for name, path in [('base', MODEL), ('+cpt', f'{BASE}/adapter_cpt'), ('+cpt+sft', f'{BASE}/adapter_sft')]:
    if name in results: continue                # resume after a disconnect
    model, tok = FastLanguageModel.from_pretrained(path, max_seq_length=3072, **LOAD); FastLanguageModel.for_inference(model)
    r = {'ppl': {s: hl_lib.perplexity(model, tok, [d['text'] for d in held if d['source'] == s]) for s in sources}}
    r['mcq_plain'] = hl_lib.mcq_accuracy(model, tok, mcq, 'plain'); r['mcq_chat'] = hl_lib.mcq_accuracy(model, tok, mcq, 'chat')
    r['gen'] = [hl_lib.generate(model, tok, v['prompt'], mode='chat' if name == '+cpt+sft' else 'plain') for v in vign[:12]]
    results[name] = r; json.dump(results, open(f'{BASE}/eval_results.json', 'w'), indent=1)
    print(name, r['ppl'], r['mcq_plain'], r['mcq_chat'])
    del model, tok; gc.collect(); torch.cuda.empty_cache()'''),
 ("code", '''import matplotlib.pyplot as plt, numpy as np
names = list(results); w = 0.25
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for i, n in enumerate(names): ax[0].bar(np.arange(len(sources)) + i * w, [results[n]['ppl'][s] for s in sources], w, label=n)
ax[0].set_xticks(np.arange(len(sources)) + w); ax[0].set_xticklabels(sources); ax[0].set_ylabel('perplexity (lower is better)'); ax[0].set_title('Held-out perplexity'); ax[0].legend()
tasks = ['MedQA', 'MedMCQA', 'PubMedQA']
for i, n in enumerate(names): ax[1].bar(np.arange(3) + i * w, [max(results[n]['mcq_plain'][t], results[n]['mcq_chat'][t]) for t in tasks], w, label=n)
ax[1].set_xticks(np.arange(3) + w); ax[1].set_xticklabels(tasks); ax[1].set_ylabel('accuracy (best of plain/chat prompt)'); ax[1].set_title('Zero-shot MCQ accuracy'); ax[1].legend()
plt.tight_layout(); plt.savefig(f'{BASE}/eval_summary.png', dpi=150); plt.show()
print('| model | ' + ' | '.join(f'PPL {s}' for s in sources) + ' | ' + ' | '.join(tasks) + ' |')
for n in names: print(f"| {n} | " + ' | '.join(f"{results[n]['ppl'][s]:.2f}" for s in sources) + ' | ' + ' | '.join(f"{max(results[n]['mcq_plain'][t], results[n]['mcq_chat'][t]):.3f}" for t in tasks) + ' |')'''),
 ("md", "## GP vignettes side by side\nRead these yourself. Do the answers name plausible conditions, red-flag symptoms, and sensible next steps? Note where the base model rambles and where the tuned one is structured. Chance is 25% for MedQA/MedMCQA (4 options) and about 33% for PubMedQA (the majority class 'yes' is about 55%)."),
 ("code", '''for i, v in enumerate(vign[:12]):
    print('=' * 100); print('PATIENT:', v['prompt'])
    for n in names: print(f'--- {n}:', results[n]['gen'][i][:600].replace('\\n', ' '))'''),
]

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
    for name, cells in (("01_cpt", NB1), ("02_sft", NB2), ("03_eval", NB3)):
        json.dump(nb(cells), open(f"notebooks/{name}.ipynb", "w"), indent=1)
    print("wrote notebooks/")
