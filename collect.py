"""Collect open health text for domain fine-tuning. Stdlib + pyarrow (for Hugging Face parquet).
  python collect.py medlineplus | moh | pmc [N] | hf | --selftest
Writes data/raw/<source>.jsonl  {id, source, title, url, license, text}.
Only sources whose terms allow this: MedlinePlus (public domain), PMC open-data bucket (AWS bulk channel, per-article licence),
Ghana MoH guideline PDFs (public government documents; licence flagged), Hugging Face datasets with permissive licences.
"""
import concurrent.futures as cf, html, io, json, os, random, re, subprocess, sys, time, urllib.request, zipfile
import xml.etree.ElementTree as ET

UA = "Mozilla/5.0 (student-research-bot; NLP coursework, non-commercial)"
RAW = "data/raw"
PMC = "https://pmc-oa-opendata.s3.amazonaws.com"
PMC_OK = {"CC0", "CC BY", "CC BY-SA", "CC BY-NC", "CC BY-NC-SA"}          # ND excluded; NC flagged in the licence field
CLINICAL = re.compile(r"\b(patients?|clinical|treatment|diagnos\w*|symptom\w*|therap\w*)\b", re.I)

def get(url, timeout=60, tries=3):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))

def write(name, docs):
    os.makedirs(RAW, exist_ok=True)
    with open(f"{RAW}/{name}.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(name, len(docs), "docs,", sum(len(d["text"].split()) for d in docs), "words")

def strip_html(s):
    s = re.sub(r"</(p|li|h\d|div|br)>|<br\s*/?>", "\n", s, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()

def pmc_body(txt):
    """Drop journal/author header and the reference list from a PMC .txt article."""
    lines = txt.split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip().lower() == "abstract"),
                 next((i + 1 for i, l in enumerate(lines) if l.startswith("Publication date")), 0))
    end = max((i for i, l in enumerate(lines) if l.strip() == "References"), default=len(lines))
    return "\n".join(lines[start:end]).strip() if end > start else "\n".join(lines[start:]).strip()

def medlineplus():
    page = get("https://medlineplus.gov/xml.html").decode()
    url = re.search(r'href="(https://medlineplus.gov/xml/mplus_topics_compressed_[\d-]+\.zip)"', page).group(1)
    z = zipfile.ZipFile(io.BytesIO(get(url)))
    root = ET.fromstring(z.read(z.namelist()[0]))
    docs = []
    for t in root.iter("health-topic"):
        if t.get("language") != "English":
            continue
        text = strip_html(t.findtext("full-summary") or "")
        if len(text.split()) >= 30:
            docs.append({"id": "mplus-" + t.get("id"), "source": "medlineplus", "title": t.get("title"), "url": t.get("url"),
                         "license": "public domain (NLM); credit: Courtesy of MedlinePlus from the National Library of Medicine", "text": f"{t.get('title')}\n\n{text}"})
    write("medlineplus", docs)

def moh():
    docs = []
    for name, url in (("Ghana Standard Treatment Guidelines, 7th ed. (2017)", "https://www.moh.gov.gh/wp-content/uploads/2020/07/GHANA-STG-2017-1.pdf"),
                      ("Ghana Essential Medicines List, 7th ed. (2017)", "https://moh.gov.gh/wp-content/uploads/2020/07/GHANA-EML-2017.pdf")):
        os.makedirs(f"{RAW}/moh", exist_ok=True)
        path = f"{RAW}/moh/{url.rsplit('/', 1)[1]}"
        open(path, "wb").write(get(url, timeout=180))
        text = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True).stdout
        docs.append({"id": "moh-" + url.rsplit('/', 1)[1][:-4].lower(), "source": "moh_ghana", "title": name, "url": url,
                     "license": "Ghana Ministry of Health publication; check reuse terms", "text": re.sub(r"[ \t]+", " ", text)})
    write("moh_ghana", docs)

def pmc_one(n, rng_seed):
    """Random-ish article: first prefix after PMC<n> in the bucket, then licence/clinical filters."""
    try:
        listing = get(f"{PMC}/?list-type=2&delimiter=/&max-keys=1&start-after=PMC{n}").decode()
        prefix = re.search(r"<Prefix>(PMC[^<]+)/</Prefix>", listing).group(1)
        meta = json.loads(get(f"{PMC}/{prefix}/{prefix}.json"))
        if meta.get("license_code") not in PMC_OK or meta.get("is_historical_ocr") or meta.get("is_retracted") or not meta.get("is_pmc_openaccess"):
            return None
        text = pmc_body(get(f"{PMC}/{prefix}/{prefix}.txt").decode("utf-8", "ignore"))
        words = len(text.split())
        if not 600 <= words <= 15000 or len(CLINICAL.findall(text)) < 15:
            return None
        return {"id": prefix, "source": "pmc_oa", "title": meta.get("title"), "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{prefix.split('.')[0]}/",
                "license": meta["license_code"], "text": text}
    except Exception:
        return None

def pmc(target=15000):
    rng, seen, docs, tried = random.Random(0), set(), [], 0
    with cf.ThreadPoolExecutor(64) as ex, open(f'{RAW}/pmc_oa.partial.jsonl', 'w') as part:
        while len(docs) < target and tried < target * 8:
            batch = [rng.randint(3_000_000, 12_500_000) for _ in range(512)]
            tried += len(batch)
            for d in ex.map(lambda n: pmc_one(n, 0), batch):
                if d and d["id"] not in seen:
                    seen.add(d["id"]); docs.append(d); part.write(json.dumps(d) + "\n")
            part.flush(); print(f"tried {tried} kept {len(docs)}", flush=True)
    write("pmc_oa", docs[:target])

HF = {  # dataset -> (parquet config/split hints); all permissive per their cards (checked 2026-09-21)
    "medical_o1_sft": ("FreedomIntelligence/medical-o1-reasoning-SFT", "apache-2.0"),
    "medmcqa": ("openlifescienceai/medmcqa", "apache-2.0"),
    "pubmedqa_labeled": ("qiaojin/PubMedQA", "mit"),
    "medqa_usmle": ("GBaker/MedQA-USMLE-4-options", "cc-by-4.0"),
    "medtext": ("BI55/MedText", "cc-by-4.0"),
    "wikitext2": ("Salesforce/wikitext", "cc-by-sa-3.0; general-English control for perplexity only"),
    "medquad": ("lavita/MedQuAD", "CC BY 4.0 per original MedQuAD release; card has no licence field, verify"),
}

def hf():
    import pyarrow.parquet as pq
    os.makedirs(f"{RAW}/hf", exist_ok=True)
    for name, (repo, lic) in HF.items():
        if os.path.exists(f"{RAW}/hf/{name}.meta.json"):
            continue
        try:
            listing = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/parquet"))
        except Exception as e:
            print("no parquet listing for", repo, e); continue
        n = 0
        for cfg, splits in listing.items():
            if repo == "qiaojin/PubMedQA" and cfg != "pqa_labeled":
                continue
            if repo == "FreedomIntelligence/medical-o1-reasoning-SFT" and cfg != "en":
                continue
            if repo == "Salesforce/wikitext" and cfg != "wikitext-2-raw-v1":
                continue
            for split, urls in splits.items():
                for i, u in enumerate(urls):
                    out = f"{RAW}/hf/{name}__{cfg}__{split}__{i}.parquet"
                    open(out, "wb").write(get(u, timeout=300)); n += pq.read_metadata(out).num_rows
        json.dump({"repo": repo, "license": lic, "rows": n}, open(f"{RAW}/hf/{name}.meta.json", "w"))
        print(name, n, "rows", lic)

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert strip_html("<p>Hi &amp; bye</p><ul><li>a</li></ul>").startswith("Hi & bye")
        art = "JOURNAL\nx\nAbstract\nBody text here\nmore\nReferences\nSmith 2020\n"
        assert pmc_body(art) == "Abstract\nBody text here\nmore"
        assert "CC BY-NC-ND" not in PMC_OK and "CC BY" in PMC_OK
        sys.exit("selftest ok")
    mode = sys.argv[1]
    {"medlineplus": medlineplus, "moh": moh, "hf": hf}.get(mode, lambda: pmc(int(sys.argv[2]) if len(sys.argv) > 2 else 15000))()
