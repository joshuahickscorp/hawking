#!/usr/bin/env python3
# Bigger diverse capture: longer natural passages + REAL repo source (code), for a trustworthy
# cross-family operator test. Records the real MLP input (post_attn_norm) at every layer.
import json,os,glob,numpy as np,mlx.core as mx
from mlx_lm import load
BF16="workspace/campaign/records/runs/qwen38-27b/bf16"; OUT="workspace/campaign/phaseB/capture_diverse2"
def clip_words(s,n): return " ".join(s.split()[:n])
# real code from the repo (naturally diverse, distinct files)
code=[]
for f in sorted(glob.glob("crates/hawking-core/src/**/*.rs",recursive=True))[:8]+sorted(glob.glob("tools/*.py"))[:8]+sorted(glob.glob("workspace/campaign/phaseB/*.py"))[:6]:
 try:
  t=open(f).read()
  if 200<len(t): code.append(clip_words(t,220))
 except: pass
code=code[:18]
prose=[
"The Antikythera mechanism, recovered from a Roman-era shipwreck, is an ancient Greek analog computer built to predict eclipses and the positions of the planets. Its intricate bronze gears, some with teeth counted in the dozens, encoded astronomical cycles with a precision that would not be matched in Europe for well over a thousand years, and its very existence forced historians to revise their assumptions about the technological ceiling of the ancient world.",
"Deep beneath the ocean surface, hydrothermal vents spew superheated, mineral-rich water into near-freezing darkness. Around them thrive ecosystems that depend not on sunlight but on chemosynthesis, where bacteria convert hydrogen sulfide into energy. Giant tube worms, ghostly crabs, and heat-tolerant microbes form food webs entirely independent of the sun, suggesting that life could arise on worlds we once dismissed as barren.",
"The Library of Alexandria was less a single building than an idea: that all the knowledge of the known world could be gathered, copied, and cross-referenced in one place. Scholars there measured the circumference of the Earth, catalogued the stars, and edited the texts of Homer. Its gradual decline, through fire, funding cuts, and neglect, is a reminder that institutions of memory are fragile and must be actively sustained.",
"Photosynthesis is arguably the most important chemical reaction on the planet. In the chloroplasts of plants and algae, light energy splits water molecules, releasing the oxygen that fills our atmosphere and fixing carbon into the sugars that feed nearly every food chain. The process is astonishingly inefficient in raw energetic terms, yet its cumulative output over billions of years transformed a lifeless rock into a living world.",
"The construction of the transcontinental railroad reshaped a continent. Crews working from opposite coasts blasted tunnels through granite, bridged canyons, and laid track across deserts, often in brutal conditions. When the final golden spike was driven, a journey that had taken months by wagon collapsed to under a week, knitting distant markets together and accelerating the settlement, and the upheaval, of the American interior.",
"Sleep, long treated as mere downtime, turns out to be a period of intense biological housekeeping. During deep sleep the brain flushes metabolic waste, consolidates memories from the day, and recalibrates hormones that govern appetite and stress. Chronic sleep deprivation is now linked to impaired judgment, weakened immunity, and long-term disease, making rest not a luxury but a physiological necessity.",
"The domestication of wild grasses into wheat, rice, and maize was among the most consequential events in human history. By selecting for larger seeds that clung to the stalk rather than scattering, early farmers slowly rewired entire species to depend on human cultivation. In turn, reliable harvests allowed permanent settlements, dense populations, and the specialization of labor that made cities possible.",
"Auroras form when charged particles from the sun, funneled by Earth's magnetic field toward the poles, collide with gases in the upper atmosphere. Oxygen glows green and red, nitrogen glows blue and violet, and the resulting curtains of light ripple across the polar sky. What appears to be a serene spectacle is in fact the visible edge of a violent interaction between our planet and the star it orbits.",
"Glass is a peculiar material, neither fully solid nor liquid but an amorphous state in which molecules are frozen in disorder. Ancient craftsmen learned to melt sand into transparent panes long before anyone understood the physics involved. Today the same substance carries the internet as optical fiber, bends light in telescopes, and forms the screens through which much of modern life is now mediated.",
"The eradication of smallpox stands as one of medicine's greatest triumphs. A coordinated global vaccination campaign, tracking outbreaks village by village, cornered a virus that had killed hundreds of millions across recorded history. By nineteen eighty the disease existed only in laboratory freezers, proving that with enough coordination, humanity could deliberately drive a pathogen to extinction.",
"Coffee began as a shrub in the highlands of Ethiopia and spread along trade routes to become a global ritual. In the coffeehouses of seventeenth-century Europe it lubricated conversation, commerce, and revolution, earning the nickname penny universities for the ideas exchanged over a cheap cup. The humble bean quietly reorganized daily rhythms around a mild stimulant.",
"Bridges are exercises in managing invisible forces. A suspension bridge hangs its roadway from cables that transfer enormous loads to towers and anchorages, converting the downward pull of gravity into tension and compression distributed across the structure. Engineers must account for wind, temperature, and resonance, lest a gentle oscillation grow, as it once did at Tacoma Narrows, into catastrophic collapse.",
]
math=[
"Consider the quadratic equation two x squared minus four x minus six equals zero. Dividing through by two gives x squared minus two x minus three, which factors as x minus three times x plus one. The roots are therefore x equals three and x equals negative one, and their sum, negative b over a, equals two, while their product, c over a, equals negative three, consistent with Vieta's formulas.",
"To find the area under the curve y equals x squared from zero to three, we integrate. The antiderivative of x squared is x cubed over three. Evaluating from zero to three gives twenty seven over three minus zero, which equals nine. Geometrically this is the accumulated area between the parabola and the horizontal axis across that interval.",
"A geometric series with first term a and common ratio r, where the absolute value of r is less than one, converges to a over one minus r. For example, one half plus one quarter plus one eighth and so on sums to one, since a equals one half and r equals one half, giving one half over one half.",
"The probability of drawing two aces in a row from a standard deck without replacement is four over fifty two times three over fifty one. That product equals twelve over two thousand six hundred fifty two, which reduces to one over two hundred twenty one, a little under half a percent.",
"By the Pythagorean theorem, a triangle with legs of length nine and twelve has a hypotenuse whose square equals eighty one plus one hundred forty four, which is two hundred twenty five. The square root of two hundred twenty five is fifteen, so the hypotenuse measures exactly fifteen units.",
"The factorial of five is five times four times three times two times one, which equals one hundred twenty. Factorials grow explosively; ten factorial already exceeds three million, which is why they appear in counting problems where order matters, such as permutations.",
"Logarithms turn multiplication into addition. Because ten to the third is one thousand, the base ten logarithm of one thousand is three. Likewise the logarithm of a product equals the sum of the logarithms, a property that once made slide rules and log tables indispensable for calculation.",
"The derivative measures instantaneous rate of change. For the function f of x equals sine x, the derivative is cosine x, so the slope of the sine curve at zero is one, and at pi over two, where sine peaks, the slope is zero, reflecting the momentary flatness at the crest.",
"A system of two linear equations, x plus y equals ten and x minus y equals four, can be solved by addition. Adding the equations eliminates y and gives two x equals fourteen, so x equals seven, and back-substitution yields y equals three.",
"The mean of the numbers four, eight, fifteen, sixteen, and twenty three is their sum, sixty six, divided by five, which equals thirteen point two. The median, the middle value when sorted, is fifteen, illustrating how mean and median can diverge in a small sample.",
]
multilingual=[
"La revolution industrielle a transforme les societes europeennes en deplacant des millions de personnes des campagnes vers les villes. Les usines ont impose de nouveaux rythmes de travail regles par l'horloge plutot que par le soleil, et les conditions difficiles ont fini par susciter des mouvements ouvriers reclamant des journees plus courtes et des salaires plus justes.",
"El descubrimiento de la penicilina por Alexander Fleming ocurrio casi por accidente cuando noto que un moho contaminante mataba las bacterias en una placa de cultivo. Aquel hallazgo fortuito abrio la era de los antibioticos y salvo incontables vidas, aunque el uso excesivo posterior ha impulsado la aparicion de bacterias resistentes.",
"Die Entwicklung der Schriftsprache zaehlt zu den wichtigsten kulturellen Errungenschaften der Menschheit. Mit dem Schreiben konnten Gesetze, Vertraege und Geschichten ueber Generationen hinweg bewahrt werden, ohne allein auf das Gedaechtnis angewiesen zu sein, und Verwaltung sowie Handel wurden ueber grosse Entfernungen moeglich.",
"La biodiversidad de los arrecifes de coral rivaliza con la de las selvas tropicales. Miles de especies de peces, moluscos y crustaceos dependen de estas estructuras vivas, que sin embargo son extremadamente sensibles a los cambios de temperatura del agua, lo que las convierte en indicadores tempranos del calentamiento global.",
"L'exploration spatiale a commence comme une competition entre deux superpuissances mais est devenue peu a peu une entreprise collaborative. La Station spatiale internationale, assemblee en orbite par plusieurs nations, symbolise cette cooperation et sert de laboratoire pour etudier les effets de l'apesanteur sur le corps humain.",
"O ciclo da agua conecta oceanos, atmosfera e continentes. A evaporacao eleva o vapor de agua, que se condensa em nuvens e retorna como chuva ou neve, alimentando rios e aquiferos. Esse movimento continuo redistribui a agua doce pelo planeta e sustenta praticamente toda a vida terrestre.",
"Die Alpen entstanden durch die Kollision der afrikanischen und der europaeischen Kontinentalplatte, ein Prozess, der ueber Millionen von Jahren Gestein auffaltete und Gipfel emporhob. Gletscher formten spaeter die Taeler, und noch heute bewegt sich das Gebirge langsam, waehrend Erosion es unablaessig abtraegt.",
"La imprenta de Gutenberg multiplico la difusion del conocimiento al hacer posible producir libros en cantidades antes inimaginables. Textos que solo existian en unos pocos manuscritos copiados a mano pudieron circular por toda Europa, alfabetizando a nuevas clases sociales y acelerando la revolucion cientifica.",
]
instruction=[
"Explain, step by step, how to implement a least-recently-used cache with constant time lookups and updates. Describe the data structures you would combine, why a hash map alone is insufficient, and how a doubly linked list lets you evict the oldest entry in constant time when capacity is exceeded.",
"Write a clear set of instructions for safely deprecating a public API endpoint. Cover announcing the timeline, adding warning headers, providing a migration guide, monitoring remaining traffic, and choosing a final shutdown date, emphasizing how to avoid breaking downstream consumers who upgrade slowly.",
"Describe how to design a database schema for a library that tracks books, members, and loans. Specify the tables, the primary and foreign keys, and how you would enforce that a single physical copy cannot be checked out to two members at the same time.",
"Outline a procedure for debugging an intermittent failure that only appears in production. Include how to add structured logging, reproduce the load pattern, isolate the failing component, and confirm a fix without simply restarting the service and hoping the problem stays away.",
"Provide guidance on writing a good commit message. Explain why the subject line should be short and imperative, why the body should describe the reason for the change rather than the mechanics, and how referencing the issue being solved helps future maintainers understand the history.",
"Explain how to convert a recursive function that computes Fibonacci numbers into an iterative one, and then into a memoized version. Compare the time and space complexity of each approach and describe when the extra memory of memoization is justified by the speedup.",
"Describe the steps to set up automated testing for a web application, from choosing unit versus integration tests, to running them on every pull request, to gating merges on their success, and explain why flaky tests undermine the entire value of the suite.",
"Write instructions for reducing the memory footprint of a program that loads a large dataset. Cover streaming instead of loading everything at once, using compact data types, releasing objects promptly, and measuring actual usage rather than guessing where the memory goes.",
]
dialogue=[
"User: I think my sourdough starter is dead, it hasn't risen in two days. Assistant: Before giving up, try a rescue feeding. Discard all but a spoonful, then feed it equal weights of flour and lukewarm water and keep it somewhere warm around twenty five degrees. If you see even small bubbles within twelve hours, the yeast is dormant, not dead, and consistent daily feedings should revive it.",
"User: Why does my code run fast locally but slowly in production? Assistant: The usual suspects are differences in data size, network latency between services, and cold caches. Locally you probably hit a small dataset and a warm disk cache, while production queries larger tables across the network. Start by adding timing around each stage and comparing where the wall clock actually goes.",
"User: Should I learn calculus before linear algebra for machine learning? Assistant: For most practical machine learning, linear algebra is the more immediately useful foundation, since models are built on vectors and matrices. You need only a working grasp of derivatives and the chain rule for training, so a light pass through calculus alongside a solid grounding in linear algebra tends to serve people best.",
"User: My houseplant's leaves are turning yellow. Assistant: Yellowing usually points to watering problems. If the soil is soggy and the yellowing starts at the bottom, you are likely overwatering and the roots need to dry out. If the leaves are crisp and the soil is bone dry, it is the opposite. Check the top inch of soil and adjust before assuming it needs fertilizer.",
"User: How do I stay motivated on a long project? Assistant: Break it into pieces small enough that finishing one feels achievable in a single sitting, and track visible progress so you can see the accumulation. Motivation tends to follow action rather than precede it, so lowering the barrier to starting each day matters more than waiting to feel inspired.",
"User: What actually happens when I clear my browser cache? Assistant: You delete the local copies of images, scripts, and pages your browser saved to load sites faster. The next visit re-downloads everything fresh, which fixes problems caused by stale files but temporarily makes pages load a little slower until the cache refills.",
"User: Is it worth buying a mechanical keyboard? Assistant: If you type for hours a day, many people find the tactile feedback and durability worth it, and the switches let you tune feel and sound to preference. If you mostly browse or type occasionally, the benefit is more about enjoyment than productivity, so it comes down to how much you value the experience.",
"User: How do noise-cancelling headphones work? Assistant: They use small microphones to sample the ambient sound, then generate an inverted sound wave that destructively interferes with the incoming noise. This works best on steady low-frequency drones like engine hum, and less well on sudden or high-pitched sounds, which is why they excel on planes but not in noisy offices.",
]
CORPUS={"prose":prose,"code":code,"math":math,"multilingual":multilingual,"instruction":instruction,"dialogue":dialogue}
model,tok=load(BF16)
lm=model
for a in ("model","language_model"):
 if hasattr(lm,a): lm=getattr(lm,a)
layers=lm.layers if hasattr(lm,"layers") else lm.model.layers
NL=len(layers); print(f"{NL} layers",flush=True)
class Tap:
 def __init__(s,orig): s.orig=orig; s.buf=None
 def __call__(s,x,*a,**k): s.buf=np.array(x.astype(mx.float32)).reshape(-1,5120).astype(np.float16); return s.orig(x,*a,**k)
taps=[Tap(layers[L].mlp) for L in range(NL)]
for L in range(NL): layers[L].mlp=taps[L]
per_layer=[[] for _ in range(NL)]; manifest=[]; row=0
for fam,prompts in CORPUS.items():
 for pi,p in enumerate(prompts):
  ids=tok.encode(p); logits=model(mx.array([ids])); mx.eval(logits)
  n=taps[0].buf.shape[0]; split="hold" if pi>=len(prompts)-3 else "fit"
  for L in range(NL): per_layer[L].append(taps[L].buf)
  manifest.append({"family":fam,"prompt_idx":pi,"n_tokens":int(n),"split":split,"row_start":row}); row+=n
 print(f"  {fam}: {len(prompts)} prompts, {sum(m['n_tokens'] for m in manifest if m['family']==fam)} tokens",flush=True)
os.makedirs(OUT,exist_ok=True)
for L in range(NL): np.concatenate(per_layer[L]).tofile(f"{OUT}/L{L:02d}.f16")
json.dump({"total_tokens":row,"n_layers":NL,"hidden":5120,"input":"post_attn_norm",
          "families":list(CORPUS.keys()),"split_rule":"last 3 prompts/family = hold",
          "manifest":manifest},open(f"{OUT}/manifest.json","w"),indent=1)
print(f"CAPTURED {row} tokens x {NL} layers -> {OUT}",flush=True)
