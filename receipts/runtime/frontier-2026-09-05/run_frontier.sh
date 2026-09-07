#!/bin/bash
cd /Users/scammermike/Downloads/hawking/.worktrees/ascension
export PYTHONUNBUFFERED=1
export PYTHONPATH=/Users/scammermike/Downloads/hawking/.worktrees/ascension/.hcli-deps${PYTHONPATH:+:$PYTHONPATH}
export HAWKING_WALL_PROFILE=/Users/scammermike/Downloads/hawking/.worktrees/ascension/wall_frontier.jsonl
rm -f wall_frontier.jsonl frontier.log
M=$(cat mission_frontier.txt)
exec python3 -u -m hcli 1 "/mission $M" --model ./ascension_envelope.hawking.json --max-cycles 3 > frontier.log 2>&1
