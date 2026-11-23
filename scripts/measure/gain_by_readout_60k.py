"""Rule gain under each readout after 60,000 training images (MNIST, pair-STDP, full scaffold,
seeds 0-4): trained runs pair_exp_S_full_N400_seed{s} vs matched untrained runs *_frozen60k, both
scored from their original run_single.py spike counts (10,000 training images for labels/probe,
full test set). Checks whether labelling still inflates the gain after a full pass (Section 3.3: 30.0 against 16.2 points)."""
import numpy as np, statistics as st, json
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
def posthoc(Xtr,ytr,Xte,yte,C=10):
    m=np.stack([Xtr[ytr==c].mean(0) for c in range(C)],1); a=m.argmax(1); a[Xtr.sum(0)==0]=-1
    v=np.zeros((len(Xte),C))
    for c in range(C):
        k=a==c
        if k.any(): v[:,c]=Xte[:,k].sum(1)/k.sum()
    return 100*(v.argmax(1)==yte).mean()
def score(d):
    zt,ze=np.load(d+"/train_spike_counts.npz"),np.load(d+"/test_spike_counts.npz")
    Xtr,ytr=zt["train_spike_counts"].astype(float),zt["train_labels"]; Xte,yte=ze["test_spike_counts"].astype(float),ze["test_labels"]
    with threadpool_limits(1): lin=100*LogisticRegression(C=1.0,max_iter=100000).fit(Xtr,ytr).score(Xte,yte)
    return posthoc(Xtr,ytr,Xte,yte), lin
rows=[]
for s in range(5):
    tl,tn=score(f"experiments/runs/pair_exp_S_full_N400_seed{s}"); rl,rn=score(f"experiments/runs/pair_exp_S_full_N400_seed{s}_frozen60k")
    rows.append(dict(seed=s,trained_lab=tl,trained_lin=tn,random_lab=rl,random_lin=rn))
    print(f"s{s}: trained lab {tl:.1f} lin {tn:.1f} | random lab {rl:.1f} lin {rn:.1f} | gain lab {tl-rl:+.1f} lin {tn-rn:+.1f}")
m=lambda k: st.mean(r[k] for r in rows)
gl=[r['trained_lab']-r['random_lab'] for r in rows]; gn=[r['trained_lin']-r['random_lin'] for r in rows]
print(f"MEAN trained lab {m('trained_lab'):.1f} lin {m('trained_lin'):.1f}; random lab {m('random_lab'):.1f} lin {m('random_lin'):.1f}")
print(f"gain labelling {st.mean(gl):.1f}±{st.stdev(gl):.1f}  gain linear {st.mean(gn):.1f}±{st.stdev(gn):.1f}  seeds lab>lin {sum(a>b for a,b in zip(gl,gn))}/5")
print(f"readout gap trained {m('trained_lin')-m('trained_lab'):.1f}, untrained {m('random_lin')-m('random_lab'):.1f}")
json.dump(rows,open("results/gain_by_readout_60k.json","w"),indent=1)
