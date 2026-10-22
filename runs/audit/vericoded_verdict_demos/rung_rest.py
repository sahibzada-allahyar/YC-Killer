"""Re-run sorryless.dafny.check() minus the task-resolution gate (read-only use of the rung's internals) on every
Dafny baseline-accept/sorryless-reject row, so checks the gate short-circuited (layout, bans, audit, compilability)
are visible."""
import sys, os, re, json, tempfile
from concurrent.futures import ThreadPoolExecutor
REPO=str(__import__('pathlib').Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO); sys.path.insert(0, '.')
os.environ['SORRYLESS_DAFNY'] = os.path.expanduser('~/.local/dafny-4.11/dafny/dafny')
from sorryless import dafny as D
from bench.audit import vericoded_rung as vr
from common import *

def check_rest(text, frozen, name='solution.dfy', timeout=300):
    spans, v = D._layout(text, frozen)
    code = D.strip_comments(text)
    if spans is None:
        return {'status': 'SPEC_TAMPERED', 'violations': v}
    v = D._seams(text, code, spans) + D._captures(code, spans)
    if v:
        return {'status': 'SPEC_TAMPERED', 'violations': v}
    agent = D._agent_spans(text, spans)
    v = D._bans([code[s:e] for s, e in agent])
    with tempfile.TemporaryDirectory(prefix='sl-rest-') as tmp:
        open(os.path.join(tmp, name), 'w').write(text)
        log = f"csv;LogFileName={os.path.join(tmp, 'verify.csv')}"
        rc, out = D._dafny(['verify', *D.VERIFY_FLAGS, '--log-format', log, name], tmp, timeout)
        m = re.search(r'finished with (\d+) verified, (\d+) errors?', out)
        ok = rc == 0 and m is not None and m.group(2) == '0'
        v += D._audit(text, name, tmp, agent)
        if ok:
            v += [f'frozen method/lemma `{n}` was not verified' for n in D._unverified(tmp, frozen['spec'])]
        rc2, tr = D._dafny(['translate', 'py', '--no-verify', '--allow-warnings', '--output', os.path.join(tmp, 'py', 'out'), name], tmp, timeout)
        if rc2 != 0:
            errs = re.findall(r'Error: (.+)', tr) or [tr.strip()[-300:]]
            v.append('not compilable: ' + ' | '.join(errs[:3]))
    return {'status': ('CHEAT_REJECTED' if v else ('PROVED' if ok else 'UNPROVED')), 'violations': v, 'verify_ok': ok,
            'verified': int(m.group(1)) if m else None}

def go(r):
    rc = rec('dafny', r['id']); text = sol(r['file'])
    frozen, adapts = vr.dafny_adapt(rc, text)
    try:
        res = check_rest(text, frozen, f"{r['id']}.dfy")
    except Exception as e:
        res = {'status': 'ERR', 'violations': [repr(e)]}
    res['id'] = r['id']
    return res

if __name__ == '__main__':
    rows = [r for r in TARGET if r['lang'] == 'dafny']
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(go, rows))
    json.dump(res, open('rung_rest.json', 'w'), indent=0)
    for x in res:
        print(x['id'], x['status'], x.get('verified'), ' || '.join(v[:140] for v in x['violations'])[:400])
