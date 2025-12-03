# Learning to Flow by Flowing to Learn:

```
Bridging Contrastive Learning and Flow Fields for Function-Aware Generation
Team Members:Kyle Kun-Hyung Roh, Vincent Jinpeng Yip
Emails:rohk@stanford.edu, vyip23@stanford.edu
```
# 1 Motivation

Recent advances in large pretrained models for proteins (e.g., ESM-2, ProtT5) and vision (e.g.,
CLIP, DINOv2) have shown that unsupervised embeddings capture structure and similarity, but
not necessarily function. In biological settings, this distinction is critical, as peptides with near-
identical sequences can diverge drastically in mechanism, and functionally similar peptides can
appear dissimilar in sequence space. Likewise, in image generation, attributes like “smile,” “age,” or
“lighting” are often entangled and uncontrollable.

```
Existing methods, such as contrastive learning, diffusion models, and latent alignment, address
these issues only partially. Contrastive models such as CLIP, SimCLR yield semantically clustered
representations but lack controllability. Diffusion models allow controllable generation but are
computationally expensive and require stochastic sampling.
We propose Function-Contrastive Latent Fields (FCLF), a novel framework for organizing pretrained
embedding spaces by function through learned continuous vector fields. Instead of relying on noise-
driven diffusion or static latent directions, FCLF learns a smooth function-conditioned flow that
transports embeddings along functional manifolds. Our hypothesis is that this will enable controllable
transitions between functional classes. For example, flowing a toxin peptide toward a hormone-like
embedding or editing an image’s expression, while maintaining structure and realism.
```
# 2 Methods

```
We build upon the theoretical foundation of score-based and flow-matching models but introduce a
novel function-conditioned vector fieldvω(z, y), wherezis a frozen embedding from a pretrained
encoder (ESM-2 for proteins, CLIP for images) andyis a function label or attribute.
This vector field approximates the conditional score:
vω(z, y)→↑zlogp(z|y),
effectively learning how latent representations “flow” toward the manifold associated with functiony.
```
```
Contrastive Flow Objective
```
```
Given a batch of embeddings{(zi,yi)}, we define one Euler step of the flow:
̃zi=zi+ωvω(zi,yi),
and a contrastive loss applied to the flowed embeddings:
```
```
LFCLF=↓
```
## ∑

```
(i,j)→P
```
```
log
```
```
exp(sim( ̃zi,zj)/ε)
∑
kexp(sim( ̃zi,zk)/ε)
```
## ,

```
whereP={(i, j):yi=yj}and sim(a, b)= a
```
```
→b
↑a↑↑b↑.
This objective pulls together samples of the same function after applying the flow and pushes apart
others, encouraging intra-functional compactness and inter-functional separability.
```
```
Field Regularization
```
```
We impose two regularizers to enforce smooth and conservative flows:
Rcurl=↔↑z↗vω(z, y)↔^2 ,Rdiv=(↑z·vω(z, y))^2.
The final training objective is:
L=LFCLF+θ 1 Rcurl+θ 2 Rdiv.
```
```
These constraints ensure thatvωbehaves like a smooth, approximately conservative field whose
integral curves align with functional densities.
```

Integration as Flow

Once trained, the vector field defines a flow operator:

```
!t(z 0 ;y)=z 0 +
```
```
∫t
```
```
0
```
```
vω(zε,y)dε,
```
which continuously transforms embeddings from one functional manifold to another (My 1 ↘My 2 ).
This can be integrated via numerical ODE solvers such as Euler or RK4.

# 3 Intended Experiments

We plan to train FCLF on two domains:

1. ProteinBase: peptides labeled by mechanism: antimicrobial, hormone, toxin, inhibitor.
2. CelebA: face images labeled by attributes: smile, age, gender.

Baselines will include frozen pretrained embeddings (ESM-2, CLIP), latent steering vectors, and
conditional diffusion models. We expect FCLF to outperform these baselines in functional clustering
and controllability while being significantly more computationally efficient.

1. Protein Function Flow

Setup:Encode sequences with ESM-2, then label by function. Trainvωon 50k samples.
Evaluation Metrics:

- Silhouette Score and Cluster Purity of embeddings after flow.
- Physicochemical metrics (hydropathy, charge) for functional realism.
2. Image Attribute Control

Setup:Encode CelebA faces using CLIP or DINOv2 embeddings.
Goal:Flow embeddings toward target attributes (neutral↘smiling).
Evaluation Metrics:

- Fréchet Inception Distance (FID) for realism.
- Attribute classification accuracy for controllability.
- Smoothness of latent trajectories.

# 4 Team Contributions

- KKR: Theoretical design and mathematical formulation (vector field dynamics, score-
    matching interpretation). Leads protein function experiments.
- VJY: Implementation and optimization of code, scribe, and visualization of latent flows.

# References

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework
for contrastive learning of visual representations. InInternational conference on machine learning.
PmLR, 1597–1607.

Muhammad Waleed Gondal, Shruti Joshi, Nasim Rahaman, Stefan Bauer, Manuel Wuthrich, and
Bernhard Schölkopf. 2021. Function Contrastive Learning of Transferable Meta-Representations.
InProceedings of the 38th International Conference on Machine Learning (Proceedings of Machine
Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 3755–3765. https:
//proceedings.mlr.press/v139/gondal21a.html


Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa,
Maryam Fazel-Zarandi, Tom Sercu, Sal Candido, et al.2022. Language models of protein
sequences at the scale of evolution enable accurate structure prediction.BioRxiv2022 (2022),
500902.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow
matching for generative modeling.arXiv preprint arXiv:2210.02747(2022).

Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin
Chen, and Arash Vahdat. 2023. Loss-guided diffusion models for plug-and-play controllable
generation. InInternational Conference on Machine Learning. PMLR, 32483–32498.


