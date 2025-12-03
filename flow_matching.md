## FLOWMATCHING FORGENERATIVEMODELING

```
Yaron Lipman^1 ,^2 Ricky T. Q. Chen^1 Heli Ben-Hamu^2 Maximilian Nickel^1 Matt Le^1
```
(^1) Meta AI (FAIR) (^2) Weizmann Institute of Science

## ABSTRACT

```
We introduce a new paradigm for generative modeling built on Continuous
Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale.
Specifically, we present the notion of Flow Matching (FM), a simulation-free
approach for training CNFs based on regressing vector fields of fixed conditional
probability paths. Flow Matching is compatible with a general family of Gaussian
probability paths for transforming between noise and data samples—which
subsumes existing diffusion paths as specific instances. Interestingly, we find
that employing FM with diffusion paths results in a more robust and stable
alternative for training diffusion models. Furthermore, Flow Matching opens
the door to training CNFs with other, non-diffusion probability paths. An
instance of particular interest is using Optimal Transport (OT) displacement
interpolation to define the conditional probability paths. These paths are more
efficient than diffusion paths, provide faster training and sampling, and result in
better generalization. Training CNFs using Flow Matching on ImageNet leads
to consistently better performance than alternative diffusion-based methods in
terms of both likelihood and sample quality, and allows fast and reliable sample
generation using off-the-shelf numerical ODE solvers.
```
## 1 INTRODUCTION

```
Deep generative models are a class of deep learning algorithms aimed at estimating and sampling
from an unknown data distribution. The recent influx of amazing advances in generative modeling,
e.g., for image generation Ramesh et al. (2022); Rombach et al. (2022), is mostly facilitated by
the scalable and relatively stable training of diffusion-based models Ho et al. (2020); Song et al.
(2020b). However, the restriction to simple diffusion processes leads to a rather confined space of
sampling probability paths, resulting in very long training times and the need to adopt specialized
methods (e.g., Song et al. (2020a); Zhang & Chen (2022)) for efficient sampling.
```
```
In this work we consider the general and deterministic framework of Continuous Normalizing
Flows (CNFs; Chen et al. (2018)). CNFs are capable of modeling arbitrary probability path
```
```
Figure 1: Unconditional ImageNet-128 sam-
ples of a CNF trained using Flow Matching
with Optimal Transport probability paths.
```
```
and are in particular known to encompass the prob-
ability paths modeled by diffusion processes (Song
et al., 2021). However, aside from diffusion that
can be trained efficiently via,e.g., denoising score
matching (Vincent, 2011), no scalable CNF train-
ing algorithms are known. Indeed, maximum like-
lihood training (e.g., Grathwohl et al. (2018)) re-
quire expensive numerical ODE simulations, while
existing simulation-free methods either involve in-
tractable integrals (Rozen et al., 2021) or biased gra-
dients (Ben-Hamu et al., 2022).
```
```
The goal of this work is to propose Flow Matching
(FM), an efficient simulation-free approach to train-
ing CNF models, allowing the adoption of general
probability paths to supervise CNF training. Impor-
tantly, FM breaks the barriers for scalable CNF train-
ing beyond diffusion, and sidesteps the need to rea-
son about diffusion processes to directly work with
probability paths.
```
# arXiv:2210.02747v2 [cs.LG] 8 Feb 2023


In particular, we propose the Flow Matching objective (Section 3), a simple and intuitive training
objective to regress onto a target vector field that generates a desired probability path. We first
show that we can construct such target vector fields through per-example (i.e., conditional) formu-
lations. Then, inspired by denoising score matching, we show that a per-example training objective,
termed Conditional Flow Matching (CFM), provides equivalent gradients and does not require ex-
plicit knowledge of the intractable target vector field. Furthermore, we discuss a general family of
per-example probability paths (Section 4) that can be used for Flow Matching, which subsumes ex-
isting diffusion paths as special instances. Even on diffusion paths, we find that using FM provides
more robust and stable training, and achieves superior performance compared to score matching.
Furthermore, this family of probability paths also includes a particularly interesting case: the vector
field that corresponds to an Optimal Transport (OT) displacement interpolant (McCann, 1997). We
find that conditional OT paths are simpler than diffusion paths, forming straight line trajectories
whereas diffusion paths result in curved paths. These properties seem to empirically translate to
faster training, faster generation, and better performance.

We empirically validate Flow Matching and the construction via Optimal Transport paths on Im-
ageNet, a large and highly diverse image dataset. We find that we can easily train models to
achieve favorable performance in both likelihood estimation and sample quality amongst competing
diffusion-based methods. Furthermore, we find that our models produce better trade-offs between
computational cost and sample quality compared to prior methods. Figure 1 depicts selected uncon-
ditional ImageNet 128×128 samples from our model.

## 2 PRELIMINARIES: CONTINUOUSNORMALIZINGFLOWS

LetRddenote the data space with data pointsx= (x^1 ,...,xd)∈Rd. Two important objects
we use in this paper are: theprobability density pathp: [0,1]×Rd →R> 0 , which is a time
dependent^1 probability density function,i.e.,

### ∫

pt(x)dx= 1, and atime-dependent vector field,
v: [0,1]×Rd→Rd. A vector fieldvtcan be used to construct a time-dependent diffeomorphic
map, called aflow,φ: [0,1]×Rd→Rd, defined via the ordinary differential equation (ODE):

```
d
dt
```
```
φt(x) =vt(φt(x)) (1)
```
```
φ 0 (x) =x (2)
```
Previously, Chen et al. (2018) suggested modeling the vector fieldvtwith a neural network,vt(x;θ),
whereθ∈Rpare its learnable parameters, which in turn leads to a deep parametric model of the
flowφt, called aContinuous Normalizing Flow(CNF). A CNF is used to reshape a simple prior
densityp 0 (e.g., pure noise) to a more complicated one,p 1 , via the push-forward equation

```
pt= [φt]∗p 0 (3)
```
where the push-forward (or change of variables) operator∗is defined by

```
[φt]∗p 0 (x) =p 0 (φ−t^1 (x)) det
```
### [

```
∂φ−t^1
∂x
```
```
(x)
```
### ]

### . (4)

A vector fieldvtis said togeneratea probability density pathptif its flowφtsatisfies equation 3.
One practical way to test if a vector field generates a probability path is using the continuity equation,
which is a key component in our proofs, see Appendix B. We recap more information on CNFs, in
particular how to compute the probabilityp 1 (x)at an arbitrary pointx∈Rdin Appendix C.

## 3 FLOWMATCHING

Letx 1 denote a random variable distributed according to some unknown data distributionq(x 1 ). We
assume we only have access to data samples fromq(x 1 )but have no access to the density function
itself. Furthermore, we letptbe a probability path such thatp 0 =pis a simple distribution,e.g., the
standard normal distributionp(x) =N(x| 0 ,I), and letp 1 be approximately equal in distribution to
q. We will later discuss how to construct such a path. The Flow Matching objective is then designed
to match this target probability path, which will allow us to flow fromp 0 top 1.

(^1) We use subscript to denote the time parameter,e.g.,pt(x).


Given a target probability density pathpt(x)and a corresponding vector fieldut(x), which generates
pt(x), we define the Flow Matching (FM) objective as

```
LFM(θ) =Et,pt(x)‖vt(x)−ut(x)‖^2 , (5)
```
whereθdenotes the learnable parameters of the CNF vector fieldvt(as defined in Section 2),t∼
U[0,1](uniform distribution), andx∼pt(x). Simply put, the FM loss regresses the vector fieldut
with a neural networkvt. Upon reaching zero loss, the learned CNF model will generatept(x).

Flow Matching is a simple and attractive objective, but na ̈ıvely on its own, it is intractable to use in
practice since we have no prior knowledge for what an appropriateptandutare. There are many
choices of probability paths that can satisfyp 1 (x)≈q(x), and more importantly, we generally
don’t have access to a closed formutthat generates the desiredpt. In this section, we show that
we can construct bothptandutusing probability paths and vector fields that are only definedper
sample, and an appropriate method of aggregation provides the desiredptandut. Furthermore, this
construction allows us to create a much more tractable objective for Flow Matching.

3.1 CONSTRUCTINGpt,utFROM CONDITIONAL PROBABILITY PATHS AND VECTOR FIELDS

A simple way to construct a target probability path is via a mixture of simpler probability paths:
Given a particular data samplex 1 we denote bypt(x|x 1 )aconditional probability pathsuch that
it satisfiesp 0 (x|x 1 ) =p(x)at timet= 0, and we designp 1 (x|x 1 )att= 1to be a distribution
concentrated aroundx=x 1 ,e.g.,p 1 (x|x 1 ) =N(x|x 1 ,σ^2 I), a normal distribution withx 1 mean
and a sufficiently small standard deviationσ > 0. Marginalizing the conditional probability paths
overq(x 1 )give rise tothe marginal probability path

```
pt(x) =
```
### ∫

```
pt(x|x 1 )q(x 1 )dx 1 , (6)
```
where in particular at timet= 1, the marginal probabilityp 1 is a mixture distribution that closely
approximates the data distributionq,

```
p 1 (x) =
```
### ∫

```
p 1 (x|x 1 )q(x 1 )dx 1 ≈q(x). (7)
```
Interestingly, we can also define amarginal vector field, by “marginalizing” over the conditional
vector fields in the following sense (we assumept(x)> 0 for alltandx):

```
ut(x) =
```
### ∫

```
ut(x|x 1 )
```
```
pt(x|x 1 )q(x 1 )
pt(x)
```
```
dx 1 , (8)
```
whereut(·|x 1 ) :Rd→Rdis a conditional vector field that generatespt(·|x 1 ). It may not seem
apparent, but this way of aggregating the conditional vector fields actually results in the correct
vector field for modeling the marginal probability path.

Our first key observation is this:

```
The marginal vector field (equation 8) generates the marginal probability path (equation 6).
```
This provides a surprising connection between the conditional VFs (those that generate conditional
probability paths) and the marginal VF (those that generate the marginal probability path). This con-
nection allows us to break down the unknown and intractable marginal VF into simpler conditional
VFs, which are much simpler to define as these only depend on a single data sample. We formalize
this in the following theorem.

Theorem 1.Given vector fieldsut(x|x 1 )that generate conditional probability pathspt(x|x 1 ), for
any distributionq(x 1 ), the marginal vector fieldutin equation 8 generates the marginal probability
pathptin equation 6, i.e.,utandptsatisfy the continuity equation (equation 26).

The full proofs for our theorems are all provided in Appendix A. Theorem 1 can also be derived
from the Diffusion Mixture Representation Theorem in Peluchetti (2021) that provides a formula
for the marginal drift and diffusion coefficients in diffusion SDEs.


### 3.2 CONDITIONALFLOWMATCHING

Unfortunately, due to the intractable integrals in the definitions of the marginal probability path
and VF (equations 6 and 8), it is still intractable to computeut, and consequently, intractable to
na ̈ıvely compute an unbiased estimator of the original Flow Matching objective. Instead, we propose
a simpler objective, which surprisingly will result in the same optima as the original objective.
Specifically, we consider theConditional Flow Matching(CFM) objective,

```
LCFM(θ) =Et,q(x 1 ),pt(x|x 1 )
```
### ∥

```
∥vt(x)−ut(x|x 1 )
```
### ∥

### ∥^2 , (9)

wheret∼U[0,1],x 1 ∼q(x 1 ), and nowx∼pt(x|x 1 ). Unlike the FM objective, the CFM objective
allows us to easily sample unbiased estimates as long as we can efficiently sample frompt(x|x 1 )
and computeut(x|x 1 ), both of which can be easily done as they are defined on a per-sample basis.
Our second key observation is therefore:

```
The FM (equation 5) and CFM (equation 9) objectives have identical gradients w.r.t.θ.
```
That is, optimizing the CFM objective is equivalent (in expectation) to optimizing the FM objective.
Consequently, this allows us to train a CNF to generate the marginal probability pathpt—which in
particular, approximates the unknown data distributionqatt= 1 — without ever needing access to
either the marginal probability path or the marginal vector field. We simply need to design suitable
conditionalprobability paths and vector fields. We formalize this property in the following theorem.

Theorem 2. Assuming thatpt(x) > 0 for allx∈Rdandt∈[0,1], then, up to a constant
independent ofθ,LCFMandLFMare equal. Hence,∇θLFM(θ) =∇θLCFM(θ).

## 4 CONDITIONALPROBABILITYPATHS ANDVECTORFIELDS

The Conditional Flow Matching objective works with any choice of conditional probability path
and conditional vector fields. In this section, we discuss the construction ofpt(x|x 1 )andut(x|x 1 )
for a general family of Gaussian conditional probability paths. Namely, we consider conditional
probability paths of the form

```
pt(x|x 1 ) =N(x|μt(x 1 ),σt(x 1 )^2 I), (10)
```
whereμ: [0,1]×Rd→Rdis the time-dependent mean of the Gaussian distribution, whileσ:
[0,1]×R→R> 0 describes a time-dependent scalar standard deviation (std). We setμ 0 (x 1 ) = 0
andσ 0 (x 1 ) = 1, so that all conditional probability paths converge to the same standard Gaussian
noise distribution att= 0,p(x) =N(x| 0 ,I). We then setμ 1 (x 1 ) =x 1 andσ 1 (x 1 ) =σmin, which
is set sufficiently small so thatp 1 (x|x 1 )is a concentrated Gaussian distribution centered atx 1.

There is an infinite number of vector fields that generate any particular probability path (e.g., by
adding a divergence free component to the continuity equation, see equation 26), but the vast major-
ity of these is due to the presence of components that leave the underlying distribution invariant—for
instance, rotational components when the distribution is rotation-invariant—leading to unnecessary
extra compute. We decide to use the simplest vector field corresponding to a canonical transforma-
tion for Gaussian distributions. Specifically, consider the flow (conditioned onx 1 )

```
ψt(x) =σt(x 1 )x+μt(x 1 ). (11)
```
Whenxis distributed as a standard Gaussian,ψt(x)is the affine transformation that maps to a
normally-distributed random variable with meanμt(x 1 )and stdσt(x 1 ). That is to say, according to
equation 4,ψtpushes the noise distributionp 0 (x|x 1 ) =p(x)topt(x|x 1 ),i.e.,

```
[ψt]∗p(x) =pt(x|x 1 ). (12)
```
This flow then provides a vector field that generates the conditional probability path:

```
d
dt
```
```
ψt(x) =ut(ψt(x)|x 1 ). (13)
```
Reparameterizingpt(x|x 1 )in terms of justx 0 and plugging equation 13 in the CFM loss we get

```
LCFM(θ) =Et,q(x 1 ),p(x 0 )
```
### ∥

### ∥

```
∥vt(ψt(x 0 ))−
```
```
d
dt
```
```
ψt(x 0 )
```
### ∥

### ∥

### ∥

```
2
```
. (14)

Sinceψtis a simple (invertible) affine map we can use equation 13 to solve forutin a closed form.
Letf′denote the derivative with respect to time,i.e.,f′=dtdf, for a time-dependent functionf.


Theorem 3.Letpt(x|x 1 )be a Gaussian probability path as in equation 10, andψtits corresponding
flow map as in equation 11. Then, the unique vector field that definesψthas the form:

```
ut(x|x 1 ) =
```
```
σ′t(x 1 )
σt(x 1 )
```
```
(x−μt(x 1 )) +μ′t(x 1 ). (15)
```
Consequently,ut(x|x 1 )generates the Gaussian pathpt(x|x 1 ).

### 4.1 SPECIAL INSTANCES OFGAUSSIAN CONDITIONAL PROBABILITY PATHS

Our formulation is fully general for arbitrary functionsμt(x 1 )andσt(x 1 ), and we can set them to
any differentiable function satisfying the desired boundary conditions. We first discuss the special
cases that recover probability paths corresponding to previously-used diffusion processes. Since we
directly work with probability paths, we can simply depart from reasoning about diffusion processes
altogether. Therefore, in the second example below, we directly formulate a probability path based
on the Wasserstein-2 optimal transport solution as an interesting instance.

Example I: Diffusion conditional VFs. Diffusion models start with data points and gradually
add noise until it approximates pure noise. These can be formulated as stochastic processes, which
have strict requirements in order to obtain closed form representation at arbitrary timest, resulting
in Gaussian conditional probability pathspt(x|x 1 )with specific choices of meanμt(x 1 )and std
σt(x 1 )(Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). For example, the reversed
(noise→data) Variance Exploding (VE) path has the form

```
pt(x) =N(x|x 1 ,σ^21 −tI), (16)
```
whereσtis an increasing function,σ 0 = 0, andσ 1  1. Next, equation 16 provides the choices of
μt(x 1 ) =x 1 andσt(x 1 ) =σ 1 −t. Plugging these into equation 15 of Theorem 3 we get

```
ut(x|x 1 ) =−
```
```
σ′ 1 −t
σ 1 −t
```
```
(x−x 1 ). (17)
```
The reversed (noise→data) Variance Preserving (VP) diffusion path has the form

```
pt(x|x 1 ) =N(x|α 1 −tx 1 ,
```
### (

```
1 −α^21 −t
```
### )

```
I),whereαt=e−
```
(^12) T(t)
,T(t) =
∫t
0
β(s)ds, (18)
andβis the noise scale function. Equation 18 provides the choices ofμt(x 1 ) =α 1 −tx 1 and
σt(x 1 ) =

### √

```
1 −α^21 −t. Plugging these into equation 15 of Theorem 3 we get
```
```
ut(x|x 1 ) =
```
```
α′ 1 −t
1 −α^21 −t
```
```
(α 1 −tx−x 1 ) =−
```
```
T′(1−t)
2
```
### [

```
e−T(1−t)x−e−
```
(^12) T(1−t)
x 1
1 −e−T(1−t)

### ]

### . (19)

Our construction of the conditional VFut(x|x 1 )does in fact coincide with the vector field previously
used in the deterministic probability flow (Song et al. (2020b), equation 13) when restricted to these
conditional diffusion processes; see details in Appendix D. Nevertheless, combining the diffusion
conditional VF with the Flow Matching objective offers an attractive training alternative—which we
find to be more stable and robust in our experiments—to existing score matching approaches.

Another important observation is that, as these probability paths were previously derived as solu-
tions of diffusion processes, they do not actually reach a true noise distribution in finite time. In
practice,p 0 (x)is simply approximated by a suitable Gaussian distribution for sampling and likeli-
hood evaluation. Instead, our construction provides full control over the probability path, and we
can just directly setμtandσt, as we will do next.

Example II: Optimal Transport conditional VFs. An arguably more natural choice for condi-
tional probability paths is to define the mean and the std to simply change linearly in time,i.e.,

```
μt(x) =tx 1 ,andσt(x) = 1−(1−σmin)t. (20)
```
According to Theorem 3 this path is generated by the VF

```
ut(x|x 1 ) =
```
```
x 1 −(1−σmin)x
1 −(1−σmin)t
```
### , (21)


```
t= 0. 0 t=^1 / 3 t=^2 / 3 t= 1. 0
Diffusion path – conditional score function
```
```
t= 0. 0 t=^1 / 3 t=^2 / 3 t= 1. 0
OT path – conditional vector field
```
Figure 2: Compared to the diffusion path’s conditional score function, the OT path’s conditional
vector field has constant direction in time and is arguably simpler to fit with a parametric model.
Note the blue color denotes larger magnitude while red color denotes smaller magnitude.

which, in contrast to the diffusion conditional VF (equation 19), is defined for allt∈[0,1]. The
conditional flow that corresponds tout(x|x 1 )is

```
ψt(x) = (1−(1−σmin)t)x+tx 1 , (22)
```
and in this case, the CFM loss (see equations 9, 14) takes the form:

```
LCFM(θ) =Et,q(x 1 ),p(x 0 )
```
### ∥

### ∥

```
∥vt(ψt(x 0 ))−
```
### (

```
x 1 −(1−σmin)x 0
```
### )∥∥

### ∥

```
2
```
. (23)

Allowing the mean and std to change linearly not only leads to simple and intuitive paths, but it
is actually also optimal in the following sense. The conditional flowψt(x)is in fact the Optimal
Transport (OT)displacement mapbetween the two Gaussiansp 0 (x|x 1 )andp 1 (x|x 1 ). The OT
interpolant, which is a probability path, is defined to be (see Definition 1.1 in McCann (1997)):

```
pt= [(1−t)id +tψ]?p 0 (24)
```
whereψ:Rd→Rdis the OT map pushingp 0 top 1 ,iddenotes the identity map,i.e.,id(x) =x,
and(1−t)id +tψis called the OT displacement map. Example 1.7 in McCann (1997) shows, that
in our case of two Gaussians where the first is a standard one, the OT displacement map takes the
form of equation 22.

```
Diffusion OT
Figure 3: Diffusion and OT
trajectories.
```
Intuitively, particles under the OT displacement map always move
in straight line trajectories and with constant speed. Figure 3 depicts
sampling paths for the diffusion and OT conditional VFs. Inter-
estingly, we find that sampling trajectory from diffusion paths can
“overshoot” the final sample, resulting in unnecessary backtracking,
whilst the OT paths are guaranteed to stay straight.

Figure 2 compares the diffusion conditional score function (the re-
gression target in a typical diffusion methods),i.e.,∇logpt(x|x 1 )withptdefined as in equation 18,
with the OT conditional VF (equation 21). The start (p 0 ) and end (p 1 ) Gaussians are identical in
both examples. An interesting observation is that the OT VF has a constant direction in time, which
arguably leads to a simpler regression task. This property can also be verified directly from equa-
tion 21 as the VF can be written in the formut(x|x 1 ) =g(t)h(x|x 1 ). Figure 8 in the Appendix
shows a visualization of the Diffusion VF. Lastly, we note that although the conditional flow is opti-
mal, this by no means imply that the marginal VF is an optimal transport solution. Nevertheless, we
expect the marginal vector field to remain relatively simple.

## 5 RELATEDWORK

Continuous Normalizing Flows were introduced in (Chen et al., 2018) as a continuous-time version
of Normalizing Flows (seee.g., Kobyzev et al. (2020); Papamakarios et al. (2021) for an overview).
Originally, CNFs are trained with the maximum likelihood objective, but this involves expensive
ODE simulations for the forward and backward propagation, resulting in high time complexity due
to the sequential nature of ODE simulations. Although some works demonstrated the capability
of CNF generative models for image synthesis (Grathwohl et al., 2018), scaling up to very high
dimensional images is inherently difficult. A number of works attempted to regularize the ODE to
be easier to solve,e.g., using augmentation (Dupont et al., 2019), adding regularization terms (Yang
& Karniadakis, 2019; Finlay et al., 2020; Onken et al., 2021; Tong et al., 2020; Kelly et al., 2020),
or stochastically sampling the integration interval (Du et al., 2022). These works merely aim to
regularize the ODE but do not change the fundamental training algorithm.


```
SM
```
```
w/ Dif
```
```
Score matchingw/ Diffusion
```
```
FM
```
```
w/ Dif
```
```
Flow Matchingw/ Diffusion
```
```
FM
```
```
w/ OT
```
```
Flow Matchingw/ OT NFE=4 NFE=8 NFE=10 NFE=
```
Figure 4: (left) Trajectories of CNFs trained with different objectives on 2D checkerboard data. The
OT path introduces the checkerboard pattern much earlier, while FM results in more stable training.
(right) FM with OT results in more efficient sampling, solved using the midpoint scheme.

In order to speed up CNF training, some works have developed simulation-free CNF training frame-
works by explicitly designing the target probability path and the dynamics. For instance, Rozen et al.
(2021) consider a linear interpolation between the prior and the target density but involves integrals
that were difficult to estimate in high dimensions, while Ben-Hamu et al. (2022) consider general
probability paths similar to this work but suffers from biased gradients in the stochastic minibatch
regime. In contrast, the Flow Matching framework allows simulation-free training with unbiased
gradients and readily scales to very high dimensions.

Another approach to simulation-free training relies on the construction of a diffusion process to
indirectly define the target probability path (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song &
Ermon, 2019). Song et al. (2020b) shows that diffusion models are trained using denoising score
matching (Vincent, 2011), a conditional objective that provides unbiased gradients with respect to
the score matching objective. Conditional Flow Matching draws inspiration from this result, but
generalizes to matching vector fields directly. Due to the ease of scalability, diffusion models have
received increased attention, producing a variety of improvements such as loss-rescaling (Song
et al., 2021), adding classifier guidance along with architectural improvements (Dhariwal & Nichol,
2021), and learning the noise schedule (Nichol & Dhariwal, 2021; Kingma et al., 2021). However,
(Nichol & Dhariwal, 2021) and (Kingma et al., 2021) only consider a restricted setting of Gaussian
conditional paths defined by simple diffusion processes with a single parameter—in particular, it
does not include our conditional OT path. In an another line of works, (De Bortoli et al., 2021; Wang
et al., 2021; Peluchetti, 2021) proposed finite time diffusion constructions via diffusion bridges
theory resolving the approximation error incurred by infinite time denoising constructions. While
existing works make use of a connection between diffusion processes and continuous normalizing
flows with the same probability path (Maoutsa et al., 2020b; Song et al., 2020b; 2021), our work
allows us to generalize beyond the class of probability paths modeled by simple diffusion. With our
work, it is possible to completely sidestep the diffusion process construction and reason directly
with probability paths, while still retaining efficient training and log-likelihood evaluations. Lastly,
concurrently to our work (Liu et al., 2022; Albergo & Vanden-Eijnden, 2022) arrived at similar
conditional objectives for simulation-free training of CNFs, while Neklyudov et al. (2023) derived
an implicit objective whenutis assumed to be a gradient field.

## 6 EXPERIMENTS

We explore the empirical benefits of using Flow Matching on the image datasets of CIFAR-
10 (Krizhevsky et al., 2009) and ImageNet at resolutions 32, 64, and 128 (Chrabaszcz et al., 2017;
Deng et al., 2009). We also ablate the choice of diffusion path in Flow Matching, particularly be-
tween the standard variance preserving diffusion path and the optimal transport path. We discuss
how sample generation is improved by directly parameterizing the generating vector field and using
the Flow Matching objective. Lastly we show Flow Matching can also be used in the conditional
generation setting. Unless otherwise specified, we evaluate likelihood and samples from the model
usingdopri5(Dormand & Prince, 1980) at absolute and relative tolerances of 1e-5. Generated
samples can be found in the Appendix, and all implementation details are in Appendix E.


CIFAR-10 ImageNet 32× 32 ImageNet 64× 64
Model NLL↓ FID↓ NFE↓ NLL↓ FID↓ NFE↓ NLL↓ FID↓ NFE↓
Ablations
DDPM 3.12 7.48 274 3.54 6.99 262 3.32 17.36 264
Score Matching 3.16 19.94 242 3.56 5.68 178 3.40 19.74 441
ScoreFlow 3.09 20.78 428 3.55 14.14 195 3.36 24.95 601
Ours
FMw/ Diffusion 3.10 8.06 183 3.54 6.37 193 3.33 16.88 187
FMw/ OT 2.99 6.35 142 3.53 5.02 122 3.31 14.45 138

```
ImageNet 128× 128
Model NLL↓ FID↓
MGAN(Hoang et al., 2018) – 58.
PacGAN2(Lin et al., 2018) – 57.
Logo-GAN-AE(Sage et al., 2018) – 50.
Self-cond. GAN(Luˇci ́c et al., 2019) – 41.
Uncond. BigGAN(Luˇci ́c et al., 2019) – 25.
PGMGAN(Armandpour et al., 2021) – 21.
FMw/ OT 2.90 20.
```
Table 1: Likelihood (BPD), quality of generated samples (FID), and evaluation time (NFE) for the
same model trained with different methods.

```
Score Matching w/ Diffusion Flow Matching w/ Diffusion Flow Matching w/ OT
```
Figure 6: Sample paths from the same initial noise with models trained on ImageNet 64×64. The
OT path reduces noise roughly linearly, while diffusion paths visibly remove noise only towards the
end of the path. Note also the differences between the generated images.

### 6.1 DENSITYMODELING ANDSAMPLEQUALITY ONIMAGENET

We start by comparing the same model architecture,i.e., the U-Net architecture from Dhariwal &
Nichol (2021) with minimal changes, trained on CIFAR-10, and ImageNet 32/64 with different
popular diffusion-based losses: DDPM from (Ho et al., 2020), Score Matching (SM) (Song et al.,
2020b), and Score Flow (SF) (Song et al., 2021); see Appendix E.1 for exact details. Table 1 (left)
summarizes our results alongside these baselines reporting negative log-likelihood (NLL) in units
of bits per dimension (BPD), sample quality as measured by the Frechet Inception Distance (FID;
Heusel et al. (2017)), and averaged number of function evaluations (NFE) required for the adaptive
solver to reach its a prespecified numerical tolerance, averaged over 50k samples. All models are
trained using the same architecture, hyperparameter values and number of training iterations, where
baselines are allowed more iterations for better convergence. Note that these areunconditional
models. On both CIFAR-10 and ImageNet, FM-OT consistently obtains best results across all our
quantitative measures compared to competing methods. We are noticing a higher that usual FID
performance in CIFAR-10 compared to previous works (Ho et al., 2020; Song et al., 2020b; 2021)
that can possibly be explained by the fact that our used architecture was not optimized for CIFAR-10.

Secondly, Table 1 (right) compares a model trained using Flow Matching with the OT path on Ima-
geNet at resolution 128×128. Our FID is state-of-the-art with the exception of IC-GAN (Casanova
et al., 2021) which uses conditioning with a self-supervised ResNet50 model, and therefore is left
out of this table. Figures 11, 12, 13 in the Appendix show non-curated samples from these models.

```
Figure 5: Image quality during
training, ImageNet 64×64.
```
Faster training.While existing works train diffusion models
with a very high number of iterations (e.g., 1.3m and 10m it-
erations are reported by Score Flow and VDM, respectively),
we find that Flow Matching generally converges much faster.
Figure 5 shows FID curves during training of Flow Matching
and all baselines for ImageNet 64×64; FM-OT is able to lower
the FID faster and to a greater extent than the alternatives. For
ImageNet-128 Dhariwal & Nichol (2021) train for 4.36m iter-
ations with batch size 256, while FM (with 25% larger model)
used 500k iterations with batch size 1.5k,i.e., 33% less image
throughput; see Table 3 for exact details. Furthermore, the cost of sampling from a model can dras-
tically change during training for score matching, whereas the sampling cost stays constant when
training with Flow Matching (Figure 10 in Appendix).

6.2 SAMPLINGEFFICIENCY

For sampling, we first draw a random noise samplex 0 ∼ N(0,I)then computeφ 1 (x 0 )by solving
equation 1 with the trained VF,vt, on the intervalt∈[0,1]using an ODE solver. While diffusion


(^2040) NFE 60 80 100
102
101
Error
SM-DifFM-Dif
FM-OT
(^02040) NFE 60 80 100
10
20
30
40
50
FID
EulerMidpoint
RK
(^02040) NFE 60 80 100
10
20
30
40
50
FID
EulerMidpoint
RK
(^02040) NFE 60 80 100
10
20
30
40
50
FID
EulerMidpoint
RK
Error of ODE solution Flow matchingw/ OT Flow matchingw/ Diffusion Score matchingw/ Diffusion
Figure 7: Flow Matching, especially when using OT paths, allows us to use fewer evaluations for
sampling while retaining similar numerical error (left) and sample quality (right). Results are shown
for models trained on ImageNet 32×32, and numerical errors are for the midpoint scheme.
models can also be sampled through an SDE formulation, this can be highly inefficient and many
methods that propose fast samplers (e.g., Song et al. (2020a); Zhang & Chen (2022)) directly make
use of the ODE perspective (see Appendix D). In part, this is due to ODE solvers being much
more efficient—yielding lower error at similar computational costs (Kloeden et al., 2012)—and the
multitude of available ODE solver schemes. When compared to our ablation models, we find that
models trained using Flow Matching with the OT path always result in the most efficient sampler,
regardless of ODE solver, as demonstrated next.
Sample paths.We first qualitatively visualize the difference in sampling paths between diffusion
and OT. Figure 6 shows samples from ImageNet-64 models using identical random seeds, where we
find that the OT path model starts generating images sooner than the diffusion path models, where
noise dominates the image until the very last time point. We additionally depict the probability
density paths in 2D generation of a checkerboard pattern, Figure 4 (left), noticing a similar trend.
Low-cost samples. We next switch to fixed-step solvers and compare low (≤100) NFE samples
computed with the ImageNet-32 models from Table 1. In Figure 7 (left), we compare the per-pixel
MSE of low NFE solutions compared with 1000 NFE solutions (we use 256 random noise seeds),
and notice that the FM with OT model produces the best numerical error, in terms of computational
cost, requiring roughly only 60% of the NFEs to reach the same error threshold as diffusion models.
Secondly, Figure 7 (right) shows how FID changes as a result of the computational cost, where we
find FM with OT is able to achieve decent FID even at very low NFE values, producing better trade-
off between sample quality and cost compared to ablated models. Figure 4 (right) shows low-cost
sampling effects for the 2D checkerboard experiment.
6.3 CONDITIONAL SAMPLING FROM LOW-RESOLUTION IMAGES
Model FID↓ IS↑PSNR↑ SSIM↑
Reference 1.9 240.8 – –
Regression 15.2 121.1 27.9 0.
SR3(Saharia et al., 2022) 5.2 180.1 26.4 0.
FMw/ OT 3.4 200.8 24.7 0.
Table 2: Image super-resolution on the
ImageNet validation set.
Lastly, we experimented with Flow Matching for condi-
tional image generation. In particular, upsampling images
from 64×64 to 256×256. We follow the evaluation proce-
dure in (Saharia et al., 2022) and compute the FID of the
upsampled validation images; baselines include reference
(FID of original validation set), and regression. Results
are in Table 2. Upsampled image samples are shown in
Figures 14, 15 in the Appendix. FM-OT achieves simi-
lar PSNR and SSIM values to (Saharia et al., 2022) while
considerably improving on FID and IS, which as argued by (Saharia et al., 2022) is a better indication
of generation quality.

## 7 CONCLUSION

We introduced Flow Matching, a new simulation-free framework for training Continuous Normaliz-
ing Flow models, relying on conditional constructions to effortlessly scale to very high dimensions.
Furthermore, the FM framework provides an alternative view on diffusion models, and suggests
forsaking the stochastic/diffusion construction in favor of more directly specifying the probability
path, allowing us to,e.g., construct paths that allow faster sampling and/or improve generation. We
experimentally showed the ease of training and sampling when using the Flow Matching framework,
and in the future, we expect FM to open the door to allowing a multitude of probability paths (e.g.,
non-isotropic Gaussians or more general kernels altogether).


## 8 SOCIAL RESPONSIBILITY

Along side its many positive applications, image generation can also be used for harmful proposes.
Using content-controlled training sets and image validation/classification can help reduce these uses.
Furthermore, the energy demand for training large deep learning models is increasing at a rapid
pace (Amodei et al., 2018; Thompson et al., 2020), focusing on methods that are able to train using
less gradient updates / image throughput can lead to significant time and energy savings.

## REFERENCES

Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic inter-
polants.arXiv preprint arXiv:2209.15571, 2022.

Dario Amodei, Danny Hernandez, Girish SastryJack, Jack Clark, Greg Brockman, and Ilya
Sutskever. Ai and compute.https://openai.com/blog/ai-and-compute/, 2018.

Mohammadreza Armandpour, Ali Sadeghian, Chunyuan Li, and Mingyuan Zhou. Partition-guided
gans. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 5099–5109, 2021.

Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Aditya Grover, Maximilian Nickel,
Ricky T. Q. Chen, and Yaron Lipman. Matching normalizing flows and probability paths on
manifolds.arXiv preprint arXiv:2207.04711, 2022.

Arantxa Casanova, Marlene Careil, Jakob Verbeek, Michal Drozdzal, and Adriana Romero Soriano.
Instance-conditioned gan.Advances in Neural Information Processing Systems, 34:27517–27529,
2021.

Ricky T. Q. Chen. torchdiffeq, 2018. URL https://github.com/rtqichen/
torchdiffeq.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary
differential equations.Advances in neural information processing systems, 31, 2018.

Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an
alternative to the cifar datasets.arXiv preprint arXiv:1707.08819, 2017.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schr ̈odinger
bridge with applications to score-based generative modeling. (arXiv:2106.01357), Dec

2021. doi: 10.48550/arXiv.2106.01357. URLhttp://arxiv.org/abs/2106.01357.
arXiv:2106.01357 [cs, math, stat].

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hier-
archical image database. In2009 IEEE Conference on Computer Vision and Pattern Recognition,
pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis.
In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.),Advances in Neu-
ral Information Processing Systems, 2021. URLhttps://openreview.net/forum?id=
AAWuCvzaVt.

John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. Journal of
computational and applied mathematics, 6(1):19–26, 1980.

Shian Du, Yihong Luo, Wei Chen, Jian Xu, and Delu Zeng. To-flow: Efficient continuous nor-
malizing flows with temporal optimization adjoint with moving speed, 2022. URLhttps:
//arxiv.org/abs/2203.10335.

Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. Augmented neural odes. In
H. Wallach, H. Larochelle, A. Beygelzimer, F. d ́ Alche-Buc, E. Fox, and R. Gar- ́
nett (eds.),Advances in Neural Information Processing Systems, volume 32. Curran Asso-
ciates, Inc., 2019. URLhttps://proceedings.neurips.cc/paper/2019/file/
21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf.


Chris Finlay, Jorn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. How to train your ̈
neural ode: the world of jacobian and kinetic regularization. InICML, pp. 3154–3164, 2020.
URLhttp://proceedings.mlr.press/v119/finlay20a.html.

Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord:
Free-form continuous dynamics for scalable reversible generative models, 2018. URLhttps:
//arxiv.org/abs/1810.01367.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash equilibrium.Advances in
neural information processing systems, 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models.Advances in
Neural Information Processing Systems, 33:6840–6851, 2020.

Quan Hoang, Tu Dinh Nguyen, Trung Le, and Dinh Phung. Mgan: Training generative adversarial
nets with multiple generators. InInternational conference on learning representations, 2018.

Jacob Kelly, Jesse Bettencourt, Matthew J Johnson, and David K Duvenaud. Learning differential
equations that are easy to solve.Advances in Neural Information Processing Systems, 33:4370–
4380, 2020.

Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models.
In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.),Advances in Neu-
ral Information Processing Systems, 2021. URLhttps://openreview.net/forum?id=
2LdBqxc1Yv.

Peter Eris Kloeden, Eckhard Platen, and Henri Schurz.Numerical solution of SDE through computer
experiments. Springer Science & Business Media, 2012.

Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and
review of current methods.IEEE transactions on pattern analysis and machine intelligence, 43
(11):3964–3979, 2020.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
2009.

Zinan Lin, Ashish Khetan, Giulia Fanti, and Sewoong Oh. Pacgan: The power of two samples in
generative adversarial networks.Advances in neural information processing systems, 31, 2018.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and
transfer data with rectified flow.arXiv preprint arXiv:2209.03003, 2022.

Mario Luˇci ́c, Michael Tschannen, Marvin Ritter, Xiaohua Zhai, Olivier Bachem, and Sylvain Gelly.
High-fidelity image generation with fewer labels. InInternational conference on machine learn-
ing, pp. 4183–4192. PMLR, 2019.

Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of
fokker–planck equations through gradient–log–density estimation.Entropy, 22(8):802, jul 2020a.
doi: 10.3390/e22080802. URLhttps://doi.org/10.3390%2Fe22080802.

Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–
planck equations through gradient–log–density estimation.Entropy, 22(8):802, 2020b.

Robert J McCann. A convexity principle for interacting gases.Advances in mathematics, 128(1):
153–179, 1997.

Kirill Neklyudov, Daniel Severo, and Alireza Makhzani. Action matching: A variational method
for learning stochastic dynamics from samples, 2023. URLhttps://openreview.net/
forum?id=T6HPzkhaKeS.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.
InInternational Conference on Machine Learning, pp. 8162–8171. PMLR, 2021.


Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. Ot-flow: Fast and accurate contin-
uous normalizing flows via optimal transport.Proceedings of the AAAI Conference on Artificial
Intelligence, 35(10):9223–9232, May 2021. URLhttps://ojs.aaai.org/index.php/
AAAI/article/view/17113.

George Papamakarios, Eric T Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji
Lakshminarayanan. Normalizing flows for probabilistic modeling and inference.J. Mach. Learn.
Res., 22(57):1–64, 2021.

Stefano Peluchetti. Non-denoising forward-time diffusions. 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-
conditional image generation with clip latents.arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ̈
resolution image synthesis with latent diffusion models. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Noam Rozen, Aditya Grover, Maximilian Nickel, and Yaron Lipman. Moser flow: Divergence-
based generative modeling on manifolds. In A. Beygelzimer, Y. Dauphin, P. Liang, and
J. Wortman Vaughan (eds.),Advances in Neural Information Processing Systems, 2021. URL
https://openreview.net/forum?id=qGvMv3undNJ.

Alexander Sage, Eirikur Agustsson, Radu Timofte, and Luc Van Gool. Logo synthesis and manipu-
lation with clustered generative adversarial networks. InProceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pp. 5879–5888, 2018.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad
Norouzi. Image super-resolution via iterative refinement.IEEE Transactions on Pattern Analysis
and Machine Intelligence, 2022.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
learning using nonequilibrium thermodynamics. InInternational Conference on Machine Learn-
ing, pp. 2256–2265. PMLR, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv
preprint arXiv:2010.02502, 2020a.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distri-
bution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d ́ Alche-Buc, E. Fox, and R. Gar- ́
nett (eds.),Advances in Neural Information Processing Systems, volume 32. Curran Asso-
ciates, Inc., 2019. URLhttps://proceedings.neurips.cc/paper/2019/file/
3001ef257407d5a371a96dcd947c7d93-Paper.pdf.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations.arXiv preprint
arXiv:2011.13456, 2020b.

Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-
based diffusion models. InThirty-Fifth Conference on Neural Information Processing Systems,
2021.

Neil C Thompson, Kristjan Greenewald, Keeheon Lee, and Gabriel F Manso. The computational
limits of deep learning.arXiv preprint arXiv:2007.05558, 2020.

Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajectorynet:
A dynamic optimal transport network for modeling cellular dynamics. InInternational conference
on machine learning, pp. 9526–9536. PMLR, 2020.

C ́edric Villani.Optimal transport: old and new, volume 338. Springer, 2009.

Pascal Vincent. A connection between score matching and denoising autoencoders.Neural compu-
tation, 23(7):1661–1674, 2011.


Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via
schr ̈odinger bridge. (arXiv:2106.10410), Jul 2021. doi: 10.48550/arXiv.2106.10410. URL
[http://arxiv.org/abs/2106.10410.](http://arxiv.org/abs/2106.10410.) arXiv:2106.10410 [cs].

Liu Yang and George E. Karniadakis. Potential flow generator with $l2$ optimal transport regu-
larity for generative models.CoRR, abs/1908.11462, 2019. URLhttp://arxiv.org/abs/
1908.11462.

Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator.
arXiv preprint arXiv:2204.13902, 2022.


## A THEOREMPROOFS

Theorem 1.Given vector fieldsut(x|x 1 )that generate conditional probability pathspt(x|x 1 ), for
any distributionq(x 1 ), the marginal vector fieldutin equation 8 generates the marginal probability
pathptin equation 6, i.e.,utandptsatisfy the continuity equation (equation 26).

Proof.To verify this, we check thatptandutsatisfy the continuity equation (equation 26):

```
d
dt
```
```
pt(x) =
```
### ∫ (

```
d
dt
```
```
pt(x|x 1 )
```
### )

```
q(x 1 )dx 1 =−
```
### ∫

```
div
```
### (

```
ut(x|x 1 )pt(x|x 1 )
```
### )

```
q(x 1 )dx 1
```
```
=−div
```
### (∫

```
ut(x|x 1 )pt(x|x 1 )q(x 1 )dx 1
```
### )

```
=−div
```
### (

```
ut(x)pt(x)
```
### )

### ,

where in the second equality we used the fact thatut(·|x 1 )generatespt(·|x 1 ), in the last equality we
used equation 8. Furthermore, the first and third equalities are justified by assuming the integrands
satisfy the regularity conditions of the Leibniz Rule (for exchanging integration and differentiation).

Theorem 2. Assuming thatpt(x) > 0 for allx∈Rdandt∈[0,1], then, up to a constant
independent ofθ,LCFMandLFMare equal. Hence,∇θLFM(θ) =∇θLCFM(θ).

Proof.To ensure existence of all integrals and to allow the changing of integration order (by Fubini’s
Theorem) in the following we assume thatq(x)andpt(x|x 1 )are decreasing to zero at a sufficient
speed as‖x‖→∞, and thatut,vt,∇θvtare bounded.

First, using the standard bilinearity of the 2 -norm we have that

```
‖vt(x)−ut(x)‖^2 =‖vt(x)‖^2 − 2 〈vt(x),ut(x)〉+‖ut(x)‖^2
```
```
‖vt(x)−ut(x|x 1 )‖^2 =‖vt(x)‖^2 − 2 〈vt(x),ut(x|x 1 )〉+‖ut(x|x 1 )‖^2
```
Next, remember thatutis independent ofθand note that

```
Ept(x)‖vt(x)‖^2 =
```
### ∫

```
‖vt(x)‖^2 pt(x)dx=
```
### ∫

```
‖vt(x)‖^2 pt(x|x 1 )q(x 1 )dx 1 dx
```
```
=Eq(x 1 ),pt(x|x 1 )‖vt(x)‖^2 ,
```
where in the second equality we use equation 6, and in the third equality we change the order of
integration. Next,

```
Ept(x)〈vt(x),ut(x)〉=
```
### ∫ 〈

```
vt(x),
```
### ∫

```
ut(x|x 1 )pt(x|x 1 )q(x 1 )dx 1
pt(x)
```
### 〉

```
pt(x)dx
```
### =

### ∫ 〈

```
vt(x),
```
### ∫

```
ut(x|x 1 )pt(x|x 1 )q(x 1 )dx 1
```
### 〉

```
dx
```
### =

### ∫

```
〈vt(x),ut(x|x 1 )〉pt(x|x 1 )q(x 1 )dx 1 dx
```
```
=Eq(x 1 ),pt(x|x 1 )〈vt(x),ut(x|x 1 )〉,
```
where in the last equality we change again the order of integration.

Theorem 3.Letpt(x|x 1 )be a Gaussian probability path as in equation 10, andψtits corresponding
flow map as in equation 11. Then, the unique vector field that definesψthas the form:

```
ut(x|x 1 ) =
```
```
σ′t(x 1 )
σt(x 1 )
```
```
(x−μt(x 1 )) +μ′t(x 1 ). (15)
```
Consequently,ut(x|x 1 )generates the Gaussian pathpt(x|x 1 ).


Proof.For notational simplicity letwt(x) =ut(x|x 1 ). Now consider equation 1:

```
d
dt
```
```
ψt(x) =wt(ψt(x)).
```
Sinceψtis invertible (asσt(x 1 )> 0 ) we letx=ψ−^1 (y)and get

```
ψ′t(ψ−^1 (y)) =wt(y), (25)
```
where we used the apostrophe notation for the derivative to emphasis thatψ′tis evaluated atψ−^1 (y).
Now, invertingψt(x)provides

```
ψ−t^1 (y) =
```
```
y−μt(x 1 )
σt(x 1 )
```
### .

Differentiatingψtwith respect totgives

```
ψ′t(x) =σt′(x 1 )x+μ′t(x 1 ).
```
Plugging these last two equations in equation 25 we get

```
wt(y) =
```
```
σ′t(x 1 )
σt(x 1 )
```
```
(y−μt(x 1 )) +μ′t(x 1 )
```
as required.

## B THE CONTINUITY EQUATION

One method of testing if a vector fieldvtgenerates a probability pathptis the continuity equation
(Villani, 2009). It is a Partial Differential Equation (PDE) providing a necessary and sufficient
condition to ensuring that a vector fieldvtgeneratespt,

```
d
dt
```
```
pt(x) + div(pt(x)vt(x)) = 0, (26)
```
where the divergence operator,div, is defined with respect to the spatial variablex= (x^1 ,...,xd),

i.e.,div =

```
∑d
i=
```
```
∂
∂xi.
```
## C COMPUTING PROBABILITIES OF THECNFMODEL

We are given an arbitrary data pointx 1 ∈Rdand need to compute the model probability at that
point,i.e.,p 1 (x 1 ). Below we recap how this can be done covering the basic relevant ODEs, the
scaling of the divergence computation, taking into account data transformations (e.g., centering of
data), and Bits-Per-Dimension computation.

ODE for computingp 1 (x 1 ) The continuity equation with equation 1 lead to the instantaneous
change of variable (Chen et al., 2018; Ben-Hamu et al., 2022):

```
d
dt
```
```
logpt(φt(x)) + div(vt(φt(x)) = 0.
```
Integratingt∈[0,1]gives:

```
logp 1 (φ 1 (x))−logp 0 (φ 0 (x)) =−
```
### ∫ 1

```
0
```
```
div(vt(φt(x)))dt (27)
```
Therefore, the log probability can be computed together with the flow trajectory by solving the ODE:

```
d
dt
```
### [

```
φt(x)
f(t)
```
### ]

### =

### [

```
vt(φt(x))
−div(vt(φt(x)))
```
### ]

### (28)

Given initial conditions
[
φ 0 (x)
f(0)

### ]

### =

### [

```
x 0
c
```
### ]

### . (29)


the solution[φt(x),f(t)]Tis uniquely defined (up to some mild conditions on the VFvt). Denote
x 1 =φ 1 (x), and according to equation 27,

```
f(1) =c+ logp 1 (x 1 )−logp 0 (x 0 ). (30)
```
Now, we are given an arbitraryx 1 and want to computep 1 (x 1 ). For this end, we will need to solve
equation 28 in reverse. That is,

```
d
ds
```
### [

```
φ 1 −s(x)
f(1−s)
```
### ]

### =

### [

```
−v 1 −s(φ 1 −s(x))
div(v 1 −s(φ 1 −s(x)))
```
### ]

### (31)

and we solve this equation fors∈[0,1]with the initial conditions ats= 0:
[
φ 1 (x)
f(1)

### ]

### =

### [

```
x 1
0
```
### ]

### . (32)

From uniqueness of ODEs, the solution will be identical to the solution of equation 28 with initial
conditions in equation 29 wherec= logp 0 (x 0 )−logp 1 (x 1 ). This can be seen from equation 30
and settingf(1) = 0. Therefore we get that

```
f(0) = logp 0 (x 0 )−logp 1 (x 1 )
```
and consequently

```
logp 1 (x 1 ) = logp 0 (x 0 )−f(0). (33)
```
To summarize, to computep 1 (x 1 )we first solve the ODE in equation 31 with initial conditions in
equation 32, and the compute equation 33.

Unbiased estimator top 1 (x 1 ) Solving equation 31 requires computation ofdivof VFs inRd
which is costly. Grathwohl et al. (2018) suggest to replace the divergence by the (unbiased) Hutchin-
son trace estimator,

```
d
ds
```
### [

```
φ 1 −s(x)
f ̃(1−s)
```
### ]

### =

### [

```
−v 1 −s(φ 1 −s(x))
zTDv 1 −s(φ 1 −s(x))z
```
### ]

### , (34)

wherez∈Rdis a sample from a random variable such thatEzzT =I. Solving the ODE in
equation 34 exactly (in practice, with a small controlled error) with initial conditions in equation 32
leads to

```
Ez
```
### [

```
logp 0 (x 0 )−f ̃(0)
```
### ]

```
= logp 0 (x 0 )−Ez
```
### [

```
f ̃(0)−f ̃(1)
```
### ]

```
= logp 0 (x 0 )−Ez
```
### [∫ 1

```
0
```
```
zTDv 1 −s(φ 1 −s(x))z ds
```
### ]

```
= logp 0 (x 0 )−
```
### ∫ 1

```
0
```
```
Ez
```
### [

```
zTDv 1 −s(φ 1 −s(x))z
```
### ]

```
ds
```
```
= logp 0 (x 0 )−
```
### ∫ 1

```
0
```
```
div(v 1 −s(φ 1 −s(x)))ds
```
```
= logp 0 (x 0 )−(f(0)−f(1))
= logp 0 (x 0 )−(logp 0 (x 0 )−logp 1 (x 1 ))
= logp 1 (x 1 ),
```
where in the third equality we switched order of integration assuming the sufficient condition of
Fubini’s theorem hold, and in the previous to last equality we used equation 30. Therefore the
random variable

```
logp 0 (x 0 )−f ̃(0) (35)
```
is an unbiased estimator forlogp 1 (x 1 ). To summarize, for a scalable unbiased estimation ofp 1 (x 1 )
we first solve the ODE in equation 34 with initial conditions in equation 32, and then output equa-
tion 35.


Transformed data Often, before training our generative model we transform the data,e.g., we
scale and/or translate the data. Such a transformation is denoted byφ−^1 :Rd →Rdand our
generative model becomes a composition

```
ψ(x) =φ◦φ(x)
```
whereφ:Rd→Rdis the model we train. Given a prior probabilityp 0 we have that the push
forward of this probability underψ(equation 3 and equation 4) takes the form

```
p 1 (x) =ψ∗p 0 (x) =p 0 (φ−^1 (φ−^1 (x))) det
```
### [

```
Dφ−^1 (φ−^1 (x))
```
### ]

```
det
```
### [

```
Dφ−^1 (x)
```
### ]

### =

### (

```
φ∗p 0 (φ−^1 (x))
```
### )

```
det
```
### [

```
Dφ−^1 (x)
```
### ]

and therefore

```
logp 1 (x) = logφ∗p 0 (φ−^1 (x)) + log det
```
### [

```
Dφ−^1 (x)
```
### ]

### .

For imagesd=H×W× 3 and we consider a transformφthat maps each pixel value from[− 1 ,1]
to[0,256]. Therefore,
φ(y) = 2^7 (y+ 1)

and
φ−^1 (x) = 2−^7 x− 1

For this case we have
logp 1 (x) = logφ∗p 0 (φ−^1 (x))− 7 dlog 2. (36)

Bits-Per-Dimension (BPD) computation BPD is defined by

```
BPD =Ex 1
```
### [

### −

```
log 2 p 1 (x 1 )
d
```
### ]

```
=Ex 1
```
### [

### −

```
logp 1 (x 1 )
dlog 2
```
### ]

### (37)

Following equation 36 we get

### BPD =−

```
logφ∗p 0 (φ−^1 (x))
dlog 2
```
### + 7

andlogφ∗p 0 (φ−^1 (x))is approximated using the unbiased estimator in equation 35 over the trans-
formed dataφ−^1 (x 1 ). Averaging the unbiased estimator on a large test testx 1 provides a good
approximation to the test set BPD.

## D DIFFUSION CONDITIONAL VECTOR FIELDS

We derive the vector field governing the Probability Flow ODE (equation 13 in Song et al. (2020b))
for the VE and VP diffusion paths (equation 18) and note that it coincides with the conditional vector
fields we derive using Theorem 3, namely the vector fields defined in equations 16 and 19.

We start with a short primer on how to find a conditional vector field for the probability path de-
scribed by the Fokker-Planck equation, then instantiate it for the VE and VP probability paths.

Since in the diffusion literature the diffusion process runs from data at timet= 0to noise at time
t= 1, we will need the following lemma to translate the diffusion VFs to our convention oft= 0
corresponds to noise andt= 1corresponds to data:

Lemma 1.Consider a flow defined by a vector fieldut(x)generating probability density pathpt(x).
Then, the vector fieldu ̃t(x) =−u 1 −t(x)generates the pathp ̃t(x) =p 1 −t(x)when initiated from
p ̃ 0 (x) =p 1 (x).

Proof.We use the continuity equation (equation 26):

```
d
dt
```
```
p ̃t(x) =
```
```
d
dt
```
```
p 1 −t(x) =−p′ 1 −t(x)
```
```
= div(p 1 −t(x)u 1 −t(x))
=−div( ̃pt(x)(−u 1 −t(x)))
```
and thereforeu ̃t(x) =−u 1 −t(x)generatesp ̃t(x).


Conditional VFs for Fokker-Planck probability paths Consider a Stochastic Differential Equa-
tion (SDE) of the standard form
dy=ftdt+gtdw (38)
with time parametert, driftft, diffusion coefficientgt, anddwis the Wiener process. The solutionyt
to the SDE is a stochastic process,i.e., a continuous time-dependent random variable, the probability
density of which,pt(yt), is characterized by the Fokker-Planck equation:

```
dpt
dt
```
```
=−div(ftpt) +
```
```
gt^2
2
```
```
∆pt (39)
```
where∆represents the Laplace operator (iny), namelydiv∇, where∇is the gradient operator
(also iny). Rewriting this equation in the form of the continuity equation can be done as follows
(Maoutsa et al., 2020a):

```
dpt
dt
```
```
=−div
```
### (

```
ftpt−
```
```
g^2
2
```
```
∇pt
pt
```
```
pt
```
### )

```
=−div
```
### ((

```
ft−
```
```
g^2 t
2
```
```
∇logpt
```
### )

```
pt
```
### )

```
=−div
```
### (

```
wtpt
```
### )

where the vector field

```
wt=ft−
```
```
g^2 t
2
```
```
∇logpt (40)
```
satisfies the continuity equation with the probability pathpt, and therefore generatespt.

Variance Exploding (VE) path The SDE for the VE path is

```
dy=
```
### √

```
d
dt
```
```
σt^2 dw,
```
whereσ 0 = 0and increasing to infinity ast→ 1. The SDE is moving from data,y 0 , att= 0to
noise,y 1 , att= 1with the probability path

```
pt(y|y 0 ) =N(y|y 0 ,σ^2 tI).
```
The conditional VF according to equation 40 is:

```
wt(y|y 0 ) =
```
```
σt′
σt
```
```
(y−y 0 )
```
Using Lemma 1 we get that the probability path

```
p ̃t(y|y 0 ) =N(y|y 0 ,σ^21 −tI)
```
is generated by

```
w ̃t(y|y 0 ) =−
```
```
σ′ 1 −t
σ 1 −t
```
```
(y−y 0 ),
```
which coincides with equation 17.

Variance Preserving (VP) path The SDE for the VP path is

```
dy=−
```
```
T′(t)
2
```
```
y+
```
### √

```
T′(t)dw,
```
whereT(t) =

```
∫t
0 β(s)ds,t∈[0,1]. The SDE coefficients are therefore
```
```
fs(y) =−
```
```
T′(s)
2
```
```
y, gs=
```
### √

```
T′(s)
```
and

```
pt(y|y 0 ) =N(y|e−
```
(^12) T(t)
y 0 ,(1−e−T(t))I).
Plugging these choices in equation 40 we get the conditional VF
wt(y|y 0 ) =
T′(t)
2

### (

```
y−e−
```
```
1
2 T(t)y 0
1 −e−T(t)
```
```
−y
```
### )

### (41)

Using Lemma 1 to reverse the time we get the conditional VF for the reverse probability path:

```
w ̃t(y|y 0 ) =−
```
```
T′(1−t)
2
```
### (

```
y−e−
```
(^12) T(1−t)
y 0
1 −e−T(1−t)
−y

### )

### =−

```
T′(1−t)
2
```
### [

```
e−T(1−t)y−e−
```
```
1
2 T(1−t)y 0
1 −e−T(1−t)
```
### ]

### ,

which coincides with equation 19.


```
t= 0. 0 t=^1 / 3 t=^2 / 3 t= 1. 0
Diffusion path – conditional vector field
```
```
Figure 8: VP Diffusion path’s conditional vector field. Compare to Figure 2.
```
```
ScoreFlow
```
```
DDPM
```
Figure 9: Trajectories of CNFs trained with ScoreFlow (Song et al., 2021) and DDPM (Ho et al.,
2020) losses on 2D checkerboard data, using the same learning rate and other hyperparameters as
Figure 4.

## E IMPLEMENTATION DETAILS

For the 2D example we used an MLP with 5-layers of 512 neurons each, while for images we used
the UNet architecture from Dhariwal & Nichol (2021). For images, we center crop images and
resize to the appropriate dimension, whereas for the 32×32 and 64×64 resolutions we use the same
pre-processing as (Chrabaszcz et al., 2017). The three methods (FM-OT, FM-Diffusion, and SM-
Diffusion) are always trained on the same architecture, same hyper-parameters, and for the same
number of epochs.

### E.1 DIFFUSION BASELINES

Losses. We consider three options as diffusion baselines that correspond to the most popular dif-
fusion loss parametrizations (Song & Ermon, 2019; Song et al., 2021; Ho et al., 2020; Kingma et al.,
2021). We will assume general Gaussian path form of equation 10,i.e.,

```
pt(x|x 1 ) =N(x|μt(x 1 ),σ^2 t(x 1 )I).
```
Score Matching loss is

```
LSM(θ) =Et,q(x 1 ),pt(x|x 1 )λ(t)‖st(x)−∇logpt(x|x 1 )‖^2 (42)
```
```
=Et,q(x 1 ),pt(x|x 1 )λ(t)
```
### ∥

### ∥

### ∥

```
∥st(x)−
```
```
x−μt(x 1 )
σ^2 t(x 1 )
```
### ∥

### ∥

### ∥

### ∥

```
2
```
. (43)

Takingλ(t) =σ^2 t(x 1 )corresponds to the original Score Matching (SM) loss from Song & Ermon
(2019), while consideringλ(t) =β(1−t)(βis defined below) corresponds to the Score Flow (SF)
loss motivated by an NLL upper bound (Song et al., 2021);stis the learnable score function. DDPM
(Noise Matching) loss from Ho et al. (2020) (equation 14) is

```
LNM(θ) =Et,q(x 1 ),pt(x|x 1 )
```
### ∥

### ∥

### ∥

```
∥t(x)−
```
```
x−μt(x 1 )
σt(x 1 )
```
### ∥

### ∥

### ∥

### ∥

```
2
(44)
```
```
=Et,q(x 1 ),p 0 (x 0 )
```
### ∥

### ∥

```
∥t(σt(x 1 )x 0 +μt(x 1 ))−x 0
```
### ∥

### ∥

### ∥

```
2
(45)
```
wherep 0 (x) =N(x| 0 ,I)is the standard Gaussian, andtis the learnable noise function.

Diffusion path. For the diffusion path we use the standard VP diffusion (equation 19), namely,

```
μt(x 1 ) =α 1 −tx 1 , σt(x 1 ) =
```
### √

```
1 −α^21 −t, whereαt=e−
```
(^12) T(t)
, T(t) =
∫t
0
β(s)ds,


```
CIFAR10 ImageNet-32 ImageNet-64 ImageNet-
Channels 256 256 192 256
Depth 2 3 3 3
Channels multiple 1,2,2,2 1,2,2,2 1,2,3,4 1,1,2,3,
Heads 4 4 4 4
Heads Channels 64 64 64 64
Attention resolution 16 16,8 32,16,8 32,16,
Dropout 0.0 0.0 0.0 0.
Effective Batch size 256 1024 2048 1536
GPUs 2 4 16 32
Epochs 1000 200 250 571
Iterations 391k 250k 157k 500k
Learning Rate 5e-4 1e-4 1e-4 1e-
Learning Rate Scheduler Polynomial Decay Polynomial Decay Constant Polynomial Decay
Warmup Steps 45k 20k - 20k
```
```
Table 3: Hyper-parameters used for training each model
```
with, as suggested in Song et al. (2020b),β(s) =βmin+s(βmax−βmin)and consequently

```
T(s) =
```
```
∫s
```
```
0
```
```
β(r)dr=sβmin+
```
### 1

### 2

```
s^2 (βmax−βmin),
```
whereβmin= 0. 1 ,βmax= 20and time is sampled in[0, 1 −],= 10−^5 for training and likelihood
and= 10−^5 for sampling.

Sampling. Score matching samples are produced by solving the ODE (equation 1) with the vector
field

```
ut(x) =−
```
```
T′(1−t)
2
```
```
[st(x)−x]. (46)
```
DDPM samples are computed with equation 46 after setting√ st(x) = t(x)/σt, whereσt =

```
1 −α 12 −t.
```
### E.2 TRAINING&EVALUATION DETAILS

We report the hyper-parameters used in Table 3. We use full 32 bit-precision for training CIFAR
and ImageNet-32 and 16-bit mixed precision for training ImageNet-64/128/256. All models are
trained using the Adam optimizer with the following parameters:β 1 = 0. 9 ,β 2 = 0. 999 , weight
decay = 0.0, and= 1e− 8. All methods we trained (i.e., FM-OT, FM-Diffusion, SM-Diffusion)
using identical architectures, with the same parameters for the the same number of Epochs (see Table
3 for details). We use either a constant learning rate schedule or a polynomial decay schedule (see
Table 3). The polynomial decay learning rate schedule includes a warm-up phase for a specified
number of training steps. In the warm-up phase, the learning rate is linearly increased from 1 e− 8
to the peak learning rate (specified in Table 3). Once the peak learning rate is achieved, it linearly
decays the learning rate down to 1 e− 8 until the final training step.

When reporting negative log-likelihood, we dequantize using the standard uniform dequantization.
We report an importance-weighted estimate using

```
log
```
### 1

### K

### ∑K

```
k=
```
```
pt(x+uk),whereuk∼U(0,1), (47)
```
withxis in{0,... , 255}and solved att= 1with an adaptive step size solverdopri5with
atol=rtol=1e-5using thetorchdiffeq(Chen, 2018) library. Estimated values for different
values ofKare in Table 4.


```
CIFAR-10 ImageNet 32× 32 ImageNet 64× 64
```
Model K=1 K=20 K=50 K=1 K=5 K=15 K=1 K=5 K=10

Ablation
DDPM 3.24 3.14 3.12 3.62 3.57 3.54 3.36 3.33 3.32
Score Matching 3.28 3.18 3.16 3.65 3.59 3.57 3.43 3.41 3.40
ScoreFlow 3.21 3.11 3.09 3.63 3.57 3.55 3.39 3.37 3.36

Ours
FMw/ Diffusion 3.23 3.13 3.10 3.64 3.58 3.56 3.37 3.34 3.33
FMw/ OT 3.11 3.01 2.99 3.62 3.56 3.53 3.35 3.33 3.31

Table 4: Negative log-likelihood (in bits per dimension) on the test set with different values ofK
using uniform dequantization.

```
0 100 200 300 400 500
Epochs
```
```
102
```
```
103
```
```
NFE
```
```
score_dif
fm_dif
fm_ot
```
Figure 10: Function evaluations for sampling during training, for models trained on CIFAR-10 using
dopri5solver with tolerance 1 e−^5.

When computing FID/Inception scores for CIFAR10, ImageNet-32/64 we use the TensorFlow GAN
library^2. To remain comparable to Dhariwal & Nichol (2021) for ImageNet-128 we use the evalua-
tion script they include in their publicly available code repository^3.

## F ADDITIONAL TABLES AND FIGURES

(^2) https://github.com/tensorflow/gan
(^3) https://github.com/openai/guided-diffusion


Figure 11: Non-curated unconditional ImageNet-32 generated images of a CNF trained with FM-
OT.


Figure 12: Non-curated unconditional ImageNet-64 generated images of a CNF trained with FM-
OT.


Figure 13: Non-curated unconditional ImageNet-128 generated images of a CNF trained with FM-
OT.


Figure 14: Conditional generation 64× 64 → 256 ×256. Flow Matching OT upsampled images from
validation set.


Figure 15: Conditional generation 64× 64 → 256 ×256. Flow Matching OT upsampled images from
validation set.


```
NFE=10 NFE=20 NFE=40 NFE=100 NFE=10 NFE=20 NFE=40 NFE=100
```
Figure 16: Generated samples from the same initial noise, but with varying number of function
evaluations (NFE). Flow matching with OT path trained on ImageNet-128.


```
NFE=10 NFE=20 NFE=40 NFE=60 NFE=100
```
Figure 17: Generated samples from the same initial noise, but with varying number of function
evaluations (NFE). Flow matching with OT path trained on ImageNet 256×256.


