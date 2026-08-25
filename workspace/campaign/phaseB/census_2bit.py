#!/usr/bin/env python3
# G21: complete measured census of the external MLX 2-bit abliterated artifact. No folder-name trust.
import json,struct,os,glob
from collections import defaultdict
D="workspace/campaign/records/runs/qwen38-27b/abliterated-mlx-2bit/2bit"
BF="workspace/campaign/records/runs/qwen38-27b/bf16"
def st_header(path):
 with open(path,'rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
 return h
DT_BYTES={'F32':4,'F16':2,'BF16':2,'U32':4,'I32':4,'U16':2,'U8':1,'I8':1,'F64':8,'I64':8,'U64':8}
def elem(shape): 
 r=1
 for s in shape: r*=s
 return r
# gather all tensors across shards
tensors={}
for shard in sorted(glob.glob(f"{D}/*.safetensors")):
 h=st_header(shard)
 for name,meta in h.items():
  if name=='__metadata__': continue
  a,b=meta['data_offsets']; tensors[name]={'dtype':meta['dtype'],'shape':meta['shape'],'bytes':b-a}
# bf16 source shapes (for effective-bpw denominator = source elements)
bf_idx=json.load(open(f"{BF}/model.safetensors.index.json"))['weight_map']
# get source shapes by reading bf16 headers once
bf_shapes={}
for shard in set(bf_idx.values()):
 h=st_header(f"{BF}/{shard}")
 for name,meta in h.items():
  if name!='__metadata__': bf_shapes[name]=meta['shape']
def role(n):
 if 'embed_tokens' in n: return 'embed'
 if 'lm_head' in n: return 'lm_head'
 if n.endswith('norm.weight') or 'layernorm' in n or 'norm' in n and 'weight' in n: return 'norm'
 if 'self_attn' in n or 'full_attn' in n: return 'attention'
 if 'linear_attn' in n or 'conv' in n or 'in_proj' in n or 'A_log' in n or 'dt_bias' in n: return 'deltanet'
 if 'mlp.gate_proj' in n or 'mlp.up_proj' in n or 'mlp.down_proj' in n: return 'mlp'
 return 'other'
# base tensor (strip .weight/.scales/.biases) to detect quant triples
def base(n):
 for suf in ('.scales','.biases','.weight'):
  if n.endswith(suf): return n[:-len(suf)],suf[1:]
 return n,''
quant=defaultdict(dict)  # base -> {weight:..,scales:..,biases:..}
plain={}
for n,t in tensors.items():
 b,suf=base(n)
 if suf in ('weight','scales','biases'): quant[b][suf]=(n,t)
 else: plain[n]=t
# classify + compute
by_role=defaultdict(lambda:{'src_elems':0,'payload':0,'meta':0,'total':0,'count':0,'quantized':0,'bits':set(),'group':set()})
per_tensor=[]
for b,parts in quant.items():
 r=role(b)
 is_q = 'scales' in parts  # quantized if it has scales
 wname,wt=parts['weight']
 src_shape=bf_shapes.get(b+'.weight',None)
 src_e=elem(src_shape) if src_shape else None
 payload=wt['bytes']
 meta=0
 if 'scales' in parts: meta+=parts['scales'][1]['bytes']
 if 'biases' in parts: meta+=parts['biases'][1]['bytes']
 tot=payload+meta
 # infer bits + group from shapes: weight is uint32 packed; scales shape [out, in/group]
 bits=None;group=None
 if is_q and src_shape:
  out,inn=src_shape[0],src_shape[1] if len(src_shape)>1 else 1
  sc_shape=parts['scales'][1]['shape']
  if len(sc_shape)==2 and sc_shape[1]>0:
   group=inn//sc_shape[1]
  # packed uint32 count vs source: bits = (packed_u32_count*32)/src_elems
  wshape=wt['shape']; packed_u32=elem(wshape)
  bits=round(packed_u32*32/src_e,3) if src_e else None
 br=by_role[r]
 if src_e: br['src_elems']+=src_e
 br['payload']+=payload;br['meta']+=meta;br['total']+=tot;br['count']+=1
 if is_q: br['quantized']+=1; br['bits'].add(bits); br['group'].add(group)
 per_tensor.append({'name':b,'role':r,'quantized':is_q,'src_shape':src_shape,'weight_dtype':wt['dtype'],
   'payload':payload,'meta':meta,'total':tot,'inferred_bits':bits,'inferred_group':group})
# plain (norms etc.) unquantized
for n,t in plain.items():
 r=role(n); br=by_role[r]; se=elem(bf_shapes.get(n,t['shape']))
 br['src_elems']+=se; br['payload']+=t['bytes']; br['total']+=t['bytes']; br['count']+=1
print(f"{'role':10}{'#':>4}{'#q':>4}{'bits':>10}{'grp':>6}{'payloadMB':>11}{'metaMB':>9}{'totalMB':>9}{'eff_bpw':>9}")
tot_bytes=0;tot_src=0
for r in ['embed','attention','deltanet','mlp','lm_head','norm','other']:
 d=by_role.get(r)
 if not d or d['count']==0: continue
 eff=d['total']*8/d['src_elems'] if d['src_elems'] else 0
 bits=','.join(str(x) for x in sorted(b for b in d['bits'] if b)) or '-'
 grp=','.join(str(x) for x in sorted(g for g in d['group'] if g)) or '-'
 print(f"{r:10}{d['count']:>4}{d['quantized']:>4}{bits:>10}{grp:>6}{d['payload']/1e6:>11.1f}{d['meta']/1e6:>9.2f}{d['total']/1e6:>9.1f}{eff:>9.3f}")
 tot_bytes+=d['total'];tot_src+=d['src_elems']
whole_bpw=tot_bytes*8/tot_src
disk=sum(os.path.getsize(f) for f in glob.glob(f"{D}/*.safetensors"))
print(f"\nWHOLE ARTIFACT: {tot_bytes/1e9:.2f} GB accounted ({tot_src/1e9:.2f}B source elems) -> COMPLETE effective BPW = {whole_bpw:.4f}")
print(f"on-disk safetensors total: {disk/1e9:.2f} GB (accounted/{disk/1e9:.2f} match check)")
out={'role_summary':{r:{k:(list(v[k]) if isinstance(v[k],set) else v[k]) for k in v} for r,v in by_role.items()},
     'complete_effective_bpw':round(whole_bpw,4),'total_bytes':tot_bytes,'source_elems':tot_src,
     'disk_bytes':disk,'per_tensor_sample':per_tensor[:6],
     'quant_config':json.load(open(f"{D}/config.json")).get('quantization')}
json.dump(out,open("receipts/ascent-2026-08-18/G21_2BIT_CENSUS.json","w"),indent=1,default=str)
print("saved G21_2BIT_CENSUS.json")
