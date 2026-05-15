# multi_prompt_bench summary

profile: `profiles/deepseek-v2-lite-q4.m3pro18.json`  
trials/prompt: 1  
max-new-tokens: 32  
speculate: `off`

| id | prompt chars | prompt tokens | completion | dec_tps (median) | dec_tps (min) | dec_tps (max) | accept rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| p001 | 16 | 4 | 32 | 21.12 | 21.12 | 21.12 | — |
| p002 | 19 | 6 | 32 | 9.60 | 9.60 | 9.60 | — |
| p003 | 19 | 6 | 32 | 20.53 | 20.53 | 20.53 | — |
| p004 | 213 | 42 | 32 | 14.21 | 14.21 | 14.21 | — |
| p005 | 212 | 36 | 32 | 12.90 | 12.90 | 12.90 | — |
| p006 | 757 | 169 | 32 | 16.63 | 16.63 | 16.63 | — |
| p007 | 626 | 153 | 32 | 6.51 | 6.51 | 6.51 | — |
