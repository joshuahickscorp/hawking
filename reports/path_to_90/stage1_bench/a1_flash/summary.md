# multi_prompt_bench summary

profile: `/tmp/profile_a1_flash.json`  
trials/prompt: 3  
max-new-tokens: 96  
speculate: `off`

| id | prompt chars | prompt tokens | completion | dec_tps (median) | dec_tps (min) | dec_tps (max) | accept rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| p001 | 16 | 4 | 96 | 14.40 | 12.91 | 16.52 | — |
| p002 | 19 | 6 | 96 | 18.36 | 17.24 | 20.57 | — |
| p003 | 19 | 6 | 96 | 18.80 | 15.63 | 18.80 | — |
| p004 | 213 | 42 | 96 | 17.62 | 16.59 | 19.95 | — |
| p005 | 212 | 36 | 96 | 18.82 | 18.20 | 20.21 | — |
| p006 | 757 | 169 | 96 | 10.12 | 6.40 | 15.74 | — |
| p007 | 626 | 153 | 67 | 13.13 | 9.28 | 16.94 | — |
