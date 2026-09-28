import json,re,sys
f=sys.argv[1]; wcell=int(sys.argv[2])
otext=open(f+'.outputs.txt').read()
nb=json.load(open('../../'+f))
w=''.join(nb['cells'][wcell]['source']).replace('−','-')
numre=re.compile(r'(?<![\w.])[-+]?\d[\d,]*\.?\d*')
toks=[]
for m in numre.finditer(otext):
    s=m.group().replace(',','').lstrip('+-')
    try: toks.append((float(s), len(s.split('.')[1]) if '.' in s.rstrip('.') else 0))
    except: pass
seen=set()
for m in numre.finditer(w):
    s=m.group().rstrip('.').replace(',','').lstrip('+-')
    if not s: continue
    x=float(s); dec=len(s.split('.')[1]) if '.' in s else 0
    if dec==0 and (x<=12 or 1990<=x<=2030): continue  # skip small ints and years
    key=s
    if key in seen: continue
    seen.add(key)
    tol=0.5*10**(-dec)+1e-12
    exact=any(abs(v-x)<=tol and dv>=dec for v,dv in toks)
    pct=any(abs(v*100-x)<=tol and dv>=dec+2 for v,dv in toks)
    if not exact and not pct:
        ctx=w[max(0,m.start()-70):m.end()+25].replace('\n',' ')
        print(f'{s:>10} | {ctx}')
