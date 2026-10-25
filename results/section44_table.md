**Table A.7(b).** Gate activity in the tuned network ($\theta_{-}=-70.6$, $\theta_{+}=-45.3$ mV) and in variants that change one setting, over the first $1{,}000$ training images (seed $0$): the share of neuron-steps on which the depression gate is open ($\bar u_{-}>\theta_{-}$) and on which the membrane exceeds $\theta_{+}$, and the ratio of total potentiation to total depression. At $\theta_{-}=-50$ mV there is no depression and potentiation is negligible (total $0.035$, against $83$ in the tuned network), because potentiation also needs $\bar u_{+}>\theta_{-}$. Part (a), accuracy, is in results/probefix_voltage_thresholds_n5.md.

| **(b)** setting | depression gate open (%) | membrane above $\theta_{+}$ (%) | potentiation : depression |
|---|--:|--:|--:|
| tuned | 36.2 | 0.0026 | 1 : 3,442 |
| inhibition off | 94.7 | 0.9867 | 1 : 11 |
| $\theta_{-}=-65$ mV | 5.0 | 0.0090 | 1 : 144 |
| $\theta_{-}=-60$ mV | 0.5 | 0.0184 | 1 : 5 |
| $\theta_{-}=-58$ mV | 0.1 | 0.0306 | 2.3 : 1 |
| $\theta_{-}=-50$ mV | 0.0 | 0.0204 | none |
| $\theta_{+}=-52$ mV | 35.7 | 0.0518 | 1 : 134 |
