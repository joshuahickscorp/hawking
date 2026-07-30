"""CLI for the laboratory harness authority."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from . import HARNESS_VERSION, SCHEMA
from .runner import Runner
from .spec import SPEC_SCHEMA, load_spec, validate_spec

def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    runner = Runner(spec, dry_run=args.dry_run)
    return runner.run()

def _cmd_validate(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    validate_spec(raw)
    print(json.dumps({'ok': True, 'id': raw['id'], 'stages': len(raw['stages'])}))
    return 0

def _cmd_version(_: argparse.Namespace) -> int:
    print(json.dumps({'schema': SCHEMA, 'version': HARNESS_VERSION, 'spec_schema': SPEC_SCHEMA}))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='lab_harness', description='Single laboratory harness: run declarative experiment specs.')
    sub = p.add_subparsers(dest='cmd', required=True)
    run_p = sub.add_parser('run', help='execute an experiment spec')
    run_p.add_argument('spec', help='path to experiment JSON spec')
    run_p.add_argument('--dry-run', action='store_true', help='print stages without executing')
    run_p.set_defaults(func=_cmd_run)
    val_p = sub.add_parser('validate', help='validate a spec without running')
    val_p.add_argument('spec', help='path to experiment JSON spec')
    val_p.set_defaults(func=_cmd_validate)
    ver_p = sub.add_parser('version', help='print harness version')
    ver_p.set_defaults(func=_cmd_version)
    return p

def main(argv: list[str] | None=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))
if __name__ == '__main__':
    raise SystemExit(main())
