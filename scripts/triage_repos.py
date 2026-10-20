#!/usr/bin/env python3
"""Triage a list of GitHub repos for MLX/Apple-Silicon relevance."""
import json, re, subprocess, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

URLS = """
https://github.com/huggingface/optimum
https://github.com/huggingface/accelerate
https://github.com/huggingface/kernels
https://github.com/huggingface/swift-jinja
https://github.com/nuemaan/skewadam
https://github.com/wuwangzhang1216/abliterix
https://github.com/Extraltodeus/J-Wash
https://github.com/zjunlp/KnowledgeCircuits
https://github.com/EnnengYang/AdaMerging
https://github.com/fulvian/engraft-ngram
https://github.com/zjunlp/DynamicKnowledgeCircuits
https://github.com/zjunlp/EasyDetect
https://github.com/modelscope/ms-swift
https://github.com/sgl-project/sglang
https://github.com/NVIDIA/TensorRT-LLM
https://github.com/FlashML-org/FreeToken
https://github.com/sail-sg/Adan
https://github.com/LiyuanLucasLiu/RAdam
https://github.com/YanaiEliyahu/AdasOptimizer
https://github.com/Liuhong99/Sophia
https://github.com/AGI-Arena/MARS
https://github.com/huggingface/peft
https://github.com/ConardLi/easy-dataset
https://github.com/bentoml/OpenLLM
https://github.com/ludwig-ai/ludwig
https://github.com/MakazhanAlpamys/Soup
https://github.com/oumi-ai/oumi
https://github.com/cloneofsimo/lora
https://github.com/Kiln-AI/Kiln
https://github.com/AutoArk/TinyEngram
https://github.com/uclaml/SPIN
https://github.com/opendilab/LightZero
https://github.com/DHDev0/Stochastic-muzero
https://github.com/uclaml/SPPO
https://github.com/argilla-io/distilabel
https://github.com/bespokelabsai/curator
https://github.com/datadreamer-dev/DataDreamer
https://github.com/magpie-align/magpie
https://github.com/BatsResearch/bonito
https://github.com/NVIDIA/Dataset_Synthesizer
https://github.com/inspirai/TimeChamber
https://github.com/blanyal/alpha-zero
https://github.com/nolabs-ai/deepfabric
https://github.com/dvgodoy/FineTuningLLMs
https://github.com/invergent-ai/surogate
https://github.com/InternScience/GraphGen
https://github.com/zyushun/Adam-mini
https://github.com/clovaai/AdamP
https://github.com/kyegomez/Sophia
https://github.com/zhuhanqing/APOLLO
https://github.com/jmrichardson/tuneta
https://github.com/solegalli/hyperparameter-optimization
https://github.com/solegalli/master-hyperparameter-optimization-for-tabular-learning
https://github.com/ModelCloud/GPTQModel
https://github.com/RyanCodrai/turbovec
https://github.com/pytorch/ao
https://github.com/X-LANCE/SLAM-LLM
https://github.com/DaoyuanLi2816/can-i-finetune-this
https://github.com/Guitaricet/relora
https://github.com/pandas-dev/pandas
https://github.com/ContextualAI/HALOs
https://github.com/princeton-nlp/SimPO
https://github.com/lordx64/phantom-kv
https://github.com/LMCache/LMCache
https://github.com/huawei-csl/KVarN
https://github.com/Yog-Sotho/LLM-fine-tuner
https://github.com/sunkencity999/blasphemer
https://github.com/elementmerc/senbonzakura
https://github.com/adybag14-cyber/Abliteration
https://github.com/PRITHIVSAKTHIUR/Auto-Abliteration
https://github.com/junainfinity/ZeroFuse
https://github.com/jonhardwick-spec/aegis-trainer
https://github.com/teilomillet/vauban
https://github.com/SuperMarioYL/abldrift
https://github.com/R0U5/Lethe
https://github.com/lauravoicu/agent-rule-gaming
https://github.com/gorrie/bias-study
https://github.com/p-e-w/heretic
""".split()

TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
     "User-Agent": "opencode", "X-GitHub-Api-Version": "2022-11-28"}

KW = {
    "mlx": re.compile(r"\bmlx\b|mlx-lm|mlx_lm|mlx-lora", re.I),
    "apple": re.compile(r"apple silicon|macos|mac m[1-9]|mps|metal", re.I),
    "cuda": re.compile(r"\bcuda\b|nvidia gpus?|requires a gpu|gpu only", re.I),
}


def get(url, accept="application/vnd.github.raw"):
    h = dict(H); h["Accept"] = accept
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def probe(url):
    slug = url.replace("https://github.com/", "").strip("/")
    rec = {"slug": slug}
    try:
        m = json.loads(get(f"https://api.github.com/repos/{slug}"))
        rec.update(stars=m.get("stargazers_count"), pushed=(m.get("pushed_at") or "")[:10],
                   lang=m.get("language"), archived=m.get("archived"),
                   lic=(m.get("license") or {}).get("spdx_id"))
    except Exception as e:
        rec["err"] = f"meta:{type(e).__name__}"
        return rec
    try:
        rm = get(f"https://api.github.com/repos/{slug}/readme")
        rec["readme"] = rm[:60000]
    except urllib.error.HTTPError as e:
        rec["readme"] = ""
        rec["rd_err"] = e.code
    except Exception as e:
        rec["readme"] = ""
    return rec


with ThreadPoolExecutor(max_workers=12) as ex:
    recs = list(ex.map(probe, URLS))

out = []
for r in recs:
    rm = r.get("readme", "")
    flags = {k: bool(p.search(rm)) for k, p in KW.items()}
    out.append({
        "slug": r["slug"], "stars": r.get("stars"), "pushed": r.get("pushed"),
        "lang": r.get("lang"), "lic": r.get("lic"), "archived": r.get("archived"),
        "mlx": flags["mlx"], "apple": flags["apple"], "cuda": flags["cuda"],
        "err": r.get("err") or r.get("rd_err") or "",
    })

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{'repo':52} {'stars':>7} {'pushed':10} {'mlx':>4} {'apple':>6} {'cuda':>5}  lang")
for o in sorted(out, key=lambda x: -(x["stars"] or 0)):
    print(f"{o['slug'][:52]:52} {str(o['stars']):>7} {str(o['pushed']):10} "
          f"{'Y' if o['mlx'] else '-':>4} {'Y' if o['apple'] else '-':>6} "
          f"{'Y' if o['cuda'] else '-':>5}  {o['lang']}{' ERR' if o['err'] else ''}")
