import sys, re, json, os, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'.')
from common import *
D=os.path.expanduser('~/.local/dafny-4.11/dafny/dafny')
unres=[r for r in TARGET if any('unresolved identifier' in v for v in r['violations'])]
def go(r):
    rc=rec('dafny',r['id'])
    sk="\n\n".join(rc[k].strip() for k in ('vc-preamble','vc-spec','vc-postamble'))+"\n"
    with tempfile.TemporaryDirectory() as t:
        open(f'{t}/f.dfy','w').write(sk)
        p=subprocess.run([D,'resolve','--allow-warnings','f.dfy'],cwd=t,capture_output=True,text=True)
    errs=re.findall(r'Error: (.+)', p.stdout+p.stderr)
    names=sorted(set(re.findall(r'unresolved identifier: (\w+)', ' '.join(errs))))
    other=sorted(set(e for e in errs if 'unresolved identifier' not in e))
    return {'id':r['id'],'names':names,'other':other}
with ThreadPoolExecutor(8) as ex: res=list(ex.map(go, unres))
json.dump(res, open('unres_names.json','w'), indent=0)
for x in res: print(x['id'], x['names'], x['other'][:2])
