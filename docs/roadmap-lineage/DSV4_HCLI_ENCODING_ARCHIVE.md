# DeepSeek HCLI encoding producer archive

`tools/condense/seal_deepseek_v4_hcli_encoding_contract.py` was an uncalled
one-shot admission producer. No active code imports it and no contract receipt
was emitted in this tree, so it had neither current authority nor unique
recorded evidence. Its implementation remains recoverable in Git; the live
DeepSeek runtime boundary and its independent native tests are unchanged.
