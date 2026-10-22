"""Context-dependent dataset valuation; synthetic demonstration, not market pricing.
Run: python experiment.py --output results.json
Requires Python >=3.10 and NumPy. No API calls or downloads.
"""
import argparse, hashlib, json, math, platform
from pathlib import Path
import numpy as np

NAMES=['A','B','A_copy','Noise']
CONTEXTS={'A_heavy':.9,'Balanced':.5,'B_heavy':.1}

def sample(rng,n,group=None,noise=False):
    g=rng.integers(0,2,n) if group is None else np.full(n,group)
    x=rng.normal(size=(n,3))
    score=np.where(g==0,x[:,0]+.5*x[:,2],x[:,1]-.5*x[:,2])
    y=np.where(score+.35*rng.normal(size=n)>0,1.,-1.)
    if noise: y=rng.choice([-1.,1.],n)
    # Separate linear boundaries, including intercepts, for the two groups.
    z=np.column_stack([g==0,(g==0)[:,None]*x,g==1,(g==1)[:,None]*x]).astype(float)
    return z,y

def fit(parts):
    x=np.concatenate([p[0] for p in parts]); y=np.concatenate([p[1] for p in parts])
    # Literal row duplicates are removed: replication is not extra evidence.
    xy=np.unique(np.column_stack([x,y]),axis=0);x=xy[:,:-1];y=xy[:,-1]
    return np.linalg.solve(x.T@x+np.eye(x.shape[1]),x.T@y)

def one(seed,n_test):
    rng=np.random.default_rng(seed)
    base_parts=[sample(rng,6,g) for g in (0,1)]
    base=(np.concatenate([x for x,y in base_parts]),np.concatenate([y for x,y in base_parts]))
    a=sample(rng,40,0); b=sample(rng,40,1)
    assets=[a,b,(a[0].copy(),a[1].copy()),sample(rng,40,noise=True)]
    tests=[sample(rng,n_test,g) for g in (0,1)]
    correct=[]
    for mask in range(16):
        w=fit([base]+[assets[i] for i in range(4) if mask&(1<<i)])
        correct.append(np.array([(np.where(x@w>0,1.,-1.)==y).astype(float) for x,y in tests]))
    correct=np.array(correct)
    rows={}
    for name,p in CONTEXTS.items():
        weights=np.array([p,1-p]); u=correct.mean(axis=2)@weights
        delta=np.array([u[1<<i]-u[0] for i in range(4)])
        loo=np.array([u[15]-u[15^(1<<i)] for i in range(4)])
        phi=np.zeros(4)
        for i in range(4):
            for mask in range(16):
                if mask&(1<<i):continue
                k=mask.bit_count();weight=math.factorial(k)*math.factorial(3-k)/math.factorial(4)
                phi[i]+=weight*(u[mask|(1<<i)]-u[mask])
        assert np.isclose(phi.sum(),u[15]-u[0],atol=1e-12)
        assert np.isclose(phi[0],phi[2],atol=1e-12)
        assert loo[0]==0 and loo[2]==0
        # Simultaneous bound for 4 predeclared acquisition contrasts x 3 contexts.
        # Differences lie in [-1,1]. Independent test points within/across groups.
        eps=math.sqrt(2*math.log(2*12/.05)*sum(weights**2)/n_test)
        rows[name]={'baseline_accuracy':float(u[0]),'all_accuracy':float(u[15]),
            'acquisition_gain':delta.tolist(),'leave_one_out':loo.tolist(),'shapley':phi.tolist(),
            'simultaneous_95_halfwidth':eps,
            'acquisition_lower':np.maximum(-1,delta-eps).tolist(),
            'acquisition_upper':np.minimum(1,delta+eps).tolist()}
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results.json');ap.add_argument('--repeats',type=int,default=50);ap.add_argument('--n-test',type=int,default=2000);args=ap.parse_args()
    runs=[one(s,args.n_test) for s in range(args.repeats)]
    summary={}
    for c in CONTEXTS:
        summary[c]={}
        for method in ['acquisition_gain','leave_one_out','shapley']:
            a=np.array([r[c][method] for r in runs])
            summary[c][method]={'mean':a.mean(0).tolist(),'sd_across_generated_instances':a.std(0,ddof=1).tolist()}
    reversal=sum((r['A_heavy']['acquisition_gain'][0]-r['A_heavy']['acquisition_gain'][1])*(r['B_heavy']['acquisition_gain'][0]-r['B_heavy']['acquisition_gain'][1])<0 for r in runs)
    report={'protocol':'contextual-data-valuation-v1','python':platform.python_version(),'numpy':np.__version__,'seeds':list(range(args.repeats)), 'n_test_per_group':args.n_test,'assets':NAMES,'context_group0_weights':CONTEXTS,'summary':summary,'A_B_acquisition_rank_reversals':reversal,'runs':runs}
    report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
