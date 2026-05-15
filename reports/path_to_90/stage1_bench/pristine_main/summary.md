# multi_prompt_bench summary

profile: `profiles/deepseek-v2-lite-q4.m3pro18.json`  
trials/prompt: 3  
max-new-tokens: 96  
speculate: `off`

| id | prompt chars | prompt tokens | completion | dec_tps (median) | dec_tps (min) | dec_tps (max) | accept rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| p001 | 16 | 4 | 96 | 20.15 | 18.21 | 20.42 | — |
| p002 | 19 | 6 | 96 | 20.64 | 19.44 | 21.77 | — |
| p003 | 19 | 6 | 96 | 21.88 | 20.22 | 22.33 | — |
| p004 | 213 | 42 | 96 | 20.54 | 18.73 | 20.70 | — |
| p005 | 212 | 36 | 96 | 17.08 | 6.64 | 21.26 | — |
| p006 | 757 | 169 | 96 | 18.88 | 17.66 | 20.51 | — |
| p007 | 626 | 153 | 67 | 18.72 | 18.31 | 19.13 | — |
