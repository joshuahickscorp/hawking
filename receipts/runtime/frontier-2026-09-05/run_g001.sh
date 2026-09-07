#!/bin/bash
cd /Users/scammermike/Downloads/hawking/.worktrees/ascension
export PYTHONUNBUFFERED=1
export HAWKING_WALL_PROFILE=/Users/scammermike/Downloads/hawking/.worktrees/ascension/wall_g001.jsonl
rm -f wall_g001.jsonl g001b.log
exec python3 -u -m hcli 1 "/mission Report the single largest measured limit in receipts/runtime/CENSUS_2026_09_05.md in one sentence." --model ./ascension_envelope.hawking.json --max-cycles 2 > g001b.log 2>&1
