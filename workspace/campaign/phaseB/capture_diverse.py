#!/usr/bin/env python3
# Honest foundation: capture the REAL MLP input (post_attn_norm) at every layer on a genuinely
# DIVERSE, NATURAL, multi-family corpus (no repeated paragraphs) so a cross-family held-out is valid.
import json,os,numpy as np,mlx.core as mx
from mlx_lm import load
BF16="workspace/campaign/records/runs/qwen38-27b/bf16"
OUT="workspace/campaign/phaseB/capture_diverse"
# ---- diverse corpus: distinct natural texts per family (NOT repeated) ----
CORPUS={
"prose":[
"The Amazon rainforest produces roughly twenty percent of the planet's oxygen and hosts more species than any other terrestrial ecosystem.",
"Medieval cathedrals took generations to build; the masons who laid the foundations rarely lived to see the spires completed.",
"Ocean currents redistribute heat from the equator toward the poles, moderating climates that would otherwise be far more extreme.",
"The printing press did not merely copy books faster; it standardized spelling, fixed texts, and made private silent reading ordinary.",
"Volcanic soil is unusually fertile because eruptions grind minerals fine and spread them across wide valleys over centuries.",
"Migratory birds navigate using a combination of the sun's position, star patterns, and a magnetic sense located in their eyes.",
"The invention of refrigeration reshaped cities by letting fresh food travel far from farms and letting families store it at home.",
"Coral reefs are built by tiny animals that secrete limestone skeletons, accumulating structures visible from orbit over millennia.",
],
"code":[
"def binary_search(a, x):\n    lo, hi = 0, len(a)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if a[mid] == x: return mid\n        if a[mid] < x: lo = mid+1\n        else: hi = mid-1\n    return -1",
"async function fetchJson(url) {\n  const res = await fetch(url);\n  if (!res.ok) throw new Error(res.status);\n  return res.json();\n}",
"impl<T: Clone> Stack<T> {\n    fn push(&mut self, v: T) { self.items.push(v); }\n    fn pop(&mut self) -> Option<T> { self.items.pop() }\n}",
"SELECT name, COUNT(*) AS n FROM orders GROUP BY name HAVING COUNT(*) > 3 ORDER BY n DESC;",
"for i in range(len(grid)):\n    for j in range(len(grid[0])):\n        if grid[i][j] == 1:\n            dfs(i, j)",
"const memo = new Map();\nfunction fib(n) {\n  if (n < 2) return n;\n  if (memo.has(n)) return memo.get(n);\n  const r = fib(n-1)+fib(n-2);\n  memo.set(n, r); return r;\n}",
"class Node:\n    def __init__(self, val):\n        self.val = val\n        self.left = self.right = None",
"pub fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 { let t = b; b = a % b; a = t; }\n    a\n}",
],
"math":[
"To solve 3x + 7 = 22, subtract 7 from both sides to get 3x = 15, then divide by 3 so x = 5.",
"The derivative of f(x) = x^3 - 4x is f'(x) = 3x^2 - 4, which is zero at x = plus or minus two over root three.",
"A right triangle with legs 5 and 12 has hypotenuse 13, since 25 plus 144 equals 169 and the square root of 169 is 13.",
"The sum of the first n integers is n(n+1)/2, so the first hundred integers sum to 5050.",
"If a fair die is rolled twice, the probability of two sixes is one thirty-sixth, since each roll is independent.",
"Integrating 2x with respect to x gives x squared plus a constant of integration.",
"The area of a circle of radius r is pi r squared; doubling the radius quadruples the area.",
"A matrix is invertible if and only if its determinant is nonzero.",
],
"multilingual":[
"La bibliotheque municipale ferme a vingt heures sauf le dimanche ou elle reste fermee toute la journee.",
"El sistema solar tiene ocho planetas y una gran cantidad de cuerpos menores como asteroides y cometas.",
"Die Alpen erstrecken sich ueber acht Laender und trennen Mitteleuropa vom Mittelmeerraum.",
"Il caffe espresso viene preparato facendo passare acqua calda ad alta pressione attraverso il caffe macinato.",
"O rio Amazonas transporta mais agua do que os proximos sete maiores rios juntos.",
"Tokyo wa sekai de mottomo jinko no ooi toshiken no hitotsu desu.",
"La migracion de las aves depende de la duracion del dia y de la temperatura estacional.",
"Les glaciers reculent chaque annee en raison du rechauffement des temperatures mondiales.",
],
"instruction":[
"Write a function that reverses a linked list in place and returns the new head without allocating extra nodes.",
"Summarize the causes of the fall of the Roman Empire in three sentences, focusing on economic and military factors.",
"Given a list of timestamps, group them into sessions where any gap longer than thirty minutes starts a new session.",
"Explain to a beginner why floating point arithmetic can make 0.1 plus 0.2 not exactly equal 0.3.",
"Draft a polite email declining a meeting invitation and proposing two alternative times next week.",
"Convert the following recipe from imperial to metric units and halve every quantity.",
"Describe an algorithm to detect a cycle in a directed graph and state its time complexity.",
"Outline the steps to safely roll back a database migration that has already been applied in production.",
],
"dialogue":[
"User: My laptop won't turn on. Assistant: Let's start simple. Is the charger light on when plugged in, and does anything appear on screen when you hold the power button for ten seconds?",
"User: What's the difference between TCP and UDP? Assistant: TCP guarantees ordered, reliable delivery with handshakes and retransmission; UDP just fires packets with no such guarantees, trading reliability for speed.",
"User: I keep procrastinating. Assistant: One approach is to shrink the task until starting feels trivial, commit to just two minutes, and let momentum carry you past the hardest part which is beginning.",
"User: Is it cheaper to fly or take the train? Assistant: It depends on distance and booking time; trains often win for short trips booked late, flights for long distances booked early.",
"User: How do vaccines work? Assistant: They expose your immune system to a harmless piece of a pathogen so it learns to recognize and fight the real thing quickly if you're ever exposed.",
"User: Recommend a book like Dune. Assistant: If you liked the politics and ecology, try Hyperion for scope or the Foundation series for grand scheming across empires.",
],
}
model,tok=load(BF16)
lm=model
for a in ("model","language_model"):
 if hasattr(lm,a): lm=getattr(lm,a)
layers=lm.layers if hasattr(lm,"layers") else lm.model.layers
NL=len(layers); print(f"{NL} layers loaded",flush=True)
# object-wrapper tap: record the real MLP input, then call the real mlp
class Tap:
 def __init__(s,orig): s.orig=orig; s.buf=[]
 def __call__(s,x,*a,**k):
  s.buf.append(np.array(x.astype(mx.float32)).reshape(-1,5120).astype(np.float16)); return s.orig(x,*a,**k)
taps=[]
for L in range(NL):
 t=Tap(layers[L].mlp); layers[L].mlp=t; taps.append(t)
# run each prompt (prefill forward), tagging tokens by family + a by-prompt split
manifest=[]; row=0
per_layer=[[] for _ in range(NL)]
fam_split={}  # token index -> (family, split)
for fam,prompts in CORPUS.items():
 for pi,p in enumerate(prompts):
  ids=tok.encode(p); 
  for t in taps: t.buf=[]
  import mlx.core as mx
  logits=model(mx.array([ids])); mx.eval(logits)
  n=taps[0].buf[0].shape[0]
  split="hold" if pi>=len(prompts)-2 else "fit"  # last 2 prompts/family = held (by-prompt, no leakage)
  for L in range(NL): per_layer[L].append(taps[L].buf[0])
  manifest.append({"family":fam,"prompt_idx":pi,"n_tokens":int(n),"split":split,"row_start":row})
  row+=n
 print(f"  {fam}: {sum(m['n_tokens'] for m in manifest if m['family']==fam)} tokens",flush=True)
os.makedirs(OUT,exist_ok=True)
for L in range(NL):
 np.concatenate(per_layer[L]).tofile(f"{OUT}/L{L:02d}.f16")
json.dump({"total_tokens":row,"n_layers":NL,"hidden":5120,"input":"post_attn_norm (real MLP input, object-wrapper tap)",
          "families":list(CORPUS.keys()),"split_rule":"last 2 prompts per family = hold (by-prompt, cross-prompt no leakage)",
          "manifest":manifest},open(f"{OUT}/manifest.json","w"),indent=1)
print(f"CAPTURED {row} tokens x {NL} layers -> {OUT}",flush=True)
