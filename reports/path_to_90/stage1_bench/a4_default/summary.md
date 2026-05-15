# multi_prompt_bench summary

profile: `profiles/deepseek-v2-lite-q4.m3pro18.json`  
trials/prompt: 3  
max-new-tokens: 96  
speculate: `off`

| id | prompt chars | prompt tokens | completion | dec_tps (median) | dec_tps (min) | dec_tps (max) | accept rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| p001 | 16 | 4 | 96 | 18.37 | 17.44 | 20.86 | — |
| p002 | 19 | 6 | 96 | 16.51 | 4.12 | 18.60 | — |
| p003 | 19 | 6 | 96 | 20.98 | 13.12 | 21.03 | — |
| p004 | 213 | 42 | 96 | 20.53 | 11.75 | 21.09 | — |
| p005 | 212 | 36 | 96 | 20.77 | 20.29 | 21.33 | — |
| p006 | 757 | 169 | 96 | 18.93 | 17.95 | 19.79 | — |
| p007 | 626 | 153 | 67 | 13.08 | 9.51 | 14.70 | — |
