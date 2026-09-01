import json, subprocess, sys
mode, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == 'cap':
    sys.path.insert(0, repo)
    from tools.future import orchestration as o
    result = o.invoke(sys.argv[4])
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(result, fh)
elif mode == 'shell':
    cmd = json.loads(sys.argv[4])
    ran = subprocess.run(cmd, cwd=repo)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'returncode': ran.returncode}, fh)
    raise SystemExit(ran.returncode)
else:
    raise SystemExit('unknown mode')
