import json,re,sys,html
f=sys.argv[1]; wcell=int(sys.argv[2])
nb=json.load(open('../../'+f))
out=[]
for i,c in enumerate(nb['cells']):
    for o in c.get('outputs',[]):
        if o['output_type']=='stream': out.append(''.join(o['text']))
        if 'data' in o:
            for m in ('text/plain','text/html'):
                if m in o['data']: out.append(html.unescape(re.sub('<[^>]+>',' ',''.join(o['data'][m]))))
        if o['output_type']=='error': out.append('\n'.join(o.get('traceback',[])))
otext='\n'.join(out).replace('−','-')
open(f+'.outputs.txt','w').write(otext)
numre=re.compile(r'(?<![\w.])[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?')
ovals=[]
for m in numre.finditer(otext):
    s=m.group().replace(',','')
    try: ovals.append(float(s))
    except: pass
import numpy as np
ov=np.array(sorted(set(ovals)))
w=''.join(nb['cells'][wcell]['source']).replace('−','-').replace('−','-')
res=[]
for m in numre.finditer(w):
    s=m.group().rstrip('.').replace(',','')
    if s in ('','-','+'): continue
    try: x=float(s)
    except: continue
    dec=len(s.split('.')[1]) if '.' in s else 0
    tol=0.5*10**(-dec)+1e-9
    ctx=w[max(0,m.start()-50):m.end()+30].replace('\n',' ')
    found=False
    for scale in (1,100,0.01,12,1/12):
        for sign in (1,-1):
            t=sign*x
            if np.any(np.abs(ov*scale-t)<=tol*1.0001): found=True;break
        if found:break
    res.append((s,found,ctx))
nf=[r for r in res if not r[1]]
print('total',len(res),'notfound',len(nf))
for s,fd,ctx in nf: print(repr(s),'|',ctx)
