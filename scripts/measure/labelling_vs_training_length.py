"""Neuron-labelling (Diehl & Cook) and linear-probe accuracy of the MNIST pair-STDP full-scaffold
network after 5k, 10k, 20k and 60k training images (seeds 0-4), all scored the same way: labels and
probe fitted on the frozen counts of 10,000 training images (*_pf.npz), scored on the 10,000 test images.
Section 3.3: the probe scores above labelling at every training length (16.4, 13.3, 9.3 and 7.7 points)."""
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
out={}
for L,suf in [(5000,"_flen5k"),(10000,"_flen10k"),(20000,"_flen20k"),(60000,"")]:
    lab,lin=[],[]
    for s in range(5):
        d=f"experiments/runs/pair_exp_S_full_N400_seed{s}{suf}"
        zt,ze=np.load(d+"/train_spike_counts_pf.npz"),np.load(d+"/test_spike_counts_pf.npz")
        Xtr,ytr=zt["train_spike_counts"].astype(float),zt["train_labels"]; Xte,yte=ze["test_spike_counts"].astype(float),ze["test_labels"]
        lab.append(posthoc(Xtr,ytr,Xte,yte))
        with threadpool_limits(1):
            lin.append(100*LogisticRegression(C=1.0,max_iter=100000).fit(Xtr,ytr).score(Xte,yte))
    out[L]=dict(labelling=lab,linear=lin)
    print(f"{L:6d} images: labelling {st.mean(lab):.1f}±{st.stdev(lab):.1f}  linear {st.mean(lin):.1f}±{st.stdev(lin):.1f}")
json.dump(out,open("results/labelling_vs_training_length.json","w"),indent=1)
