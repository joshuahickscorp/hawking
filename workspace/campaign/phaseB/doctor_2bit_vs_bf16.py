#!/usr/bin/env python3
# G27 (wider battery) + G28 (prime-failure logit diagnosis): bf16 vs external 2-bit. SPECIMEN STUDY (not a
# Hawking-NX runtime claim). Delta bf16->2bit is what matters.
import json,mlx.core as mx
from mlx_lm import load, generate
BF16="workspace/campaign/records/runs/qwen38-27b/bf16"
Q2="workspace/campaign/records/runs/qwen38-27b/abliterated-mlx-2bit/2bit"
BATTERY=[
 ("arith","The first three prime numbers are","2, 3"),
 ("arith","2 plus 2 equals","4"),
 ("arith","100 divided by 4 equals","25"),
 ("arith","The square root of 144 is","12"),
 ("arith","17 times 19 equals","323"),
 ("fact","The capital of Japan is","Tokyo"),
 ("fact","The chemical symbol for gold is","Au"),
 ("fact","The largest planet in the solar system is","Jupiter"),
 ("fact","Water boils at 100 degrees","Celsius"),
 ("fact","The author of Romeo and Juliet is","Shakespeare"),
 ("reason","If all cats are animals and Felix is a cat, then Felix is an","animal"),
 ("reason","A is taller than B, B is taller than C. The shortest is","C"),
 ("reason","The opposite of increase is","decrease"),
 ("code","In Python, len([1,2,3]) returns","3"),
 ("code","The Python keyword to define a function is","def"),
 ("code","In JSON, a list is enclosed in square","brackets"),
 ("inst","Complete: roses are red, violets are","blue"),
 ("inst","The plural of 'mouse' (animal) is","mice"),
 ("lang","Bonjour means hello in","French"),
 ("lang","The past tense of 'run' is","ran"),
]
def battery(model,tok):
 by={}; hits=0
 for cat,p,want in BATTERY:
  txt=generate(model,tok,prompt=p,max_tokens=10,verbose=False)
  ok=want.lower() in txt.lower(); hits+=ok
  by.setdefault(cat,[0,0]); by[cat][0]+=ok; by[cat][1]+=1
 return hits,len(BATTERY),by
def logits_at(model,tok,prompt,steps=4):
 ids=tok.encode(prompt); seq=list(ids); trace=[]
 for _ in range(steps):
  lg=model(mx.array([seq]))[0,-1]; mx.eval(lg)
  order=mx.argsort(-lg); top=[int(order[i]) for i in range(3)]
  vals=[float(lg[t]) for t in top]
  trace.append({'tok':[tok.decode([t]) for t in top],'logit':[round(v,2) for v in vals],'margin':round(vals[0]-vals[1],3)})
  seq.append(top[0])
 return trace
res={}
for path,label in [(Q2,"2bit"),(BF16,"bf16")]:
 model,tok=load(path)
 h,n,by=battery(model,tok)
 pr=logits_at(model,tok,"The first three prime numbers are",5)
 res[label]={'battery':f"{h}/{n}",'by_cat':{k:f"{v[0]}/{v[1]}" for k,v in by.items()},'prime_trace':pr}
 print(f"{label}: battery {h}/{n} | by-cat {res[label]['by_cat']}",flush=True)
 print(f"   prime greedy: {' '.join(s['tok'][0] for s in pr)!r}",flush=True)
 del model,tok; mx.clear_cache()
# prime divergence: first step where 2bit top token != bf16 top token
div=None
for i,(a,b) in enumerate(zip(res['2bit']['prime_trace'],res['bf16']['prime_trace'])):
 if a['tok'][0]!=b['tok'][0]: div=i; break
print(f"\nPRIME DIVERGENCE at step {div}:")
if div is not None:
 print(f"  bf16 -> {res['bf16']['prime_trace'][div]['tok'][0]!r} (margin {res['bf16']['prime_trace'][div]['margin']}, top3 {res['bf16']['prime_trace'][div]['tok']})")
 print(f"  2bit -> {res['2bit']['prime_trace'][div]['tok'][0]!r} (margin {res['2bit']['prime_trace'][div]['margin']}, top3 {res['2bit']['prime_trace'][div]['tok']})")
json.dump(res,open("receipts/ascent-2026-08-18/G27_G28_DOCTOR_2BIT.json","w"),indent=1)
print("saved G27_G28_DOCTOR_2BIT.json")
