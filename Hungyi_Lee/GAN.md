# GAN
[link](https://www.youtube.com/watch?v=z83Edfvgd9g)<br>
[link](https://www.youtube.com/watch?v=DQNNMiAP5lw)<br>

## Basic Idea of GAN
<p>
vector $\rhd$ *Genarator*(a neural network, i.e. a funtion) $\rhd$ image(high dimensional vector)<br>

Meanwhile, we train a *Discriminator*. Input: an image, output: scalar, the larger the better.
<p>

<br>

**Survival of the fittest**

- Moths versus Birds<br>
> Birds eat Moths.<br>
> Moths grow green to hide on trees.<br>
> Birds sharpen their vision.<br>
> Moths evolve to mimic leaves.<br>
> ...<br>

- Criminals versus Police<br>
> Criminals forge currency.<br>
> Police detect the flaws.<br>
> Criminals refine their craft.<br>
> Police upgrade their tech.<br>
> ...<br>

**Generative Adversarial Network.**

## Algorithm

- Initialize generator and discriminator $G,D$
- In each training iteration

    Step 1: Fix genrator $G$, and update discrimatior $D$<br>

        Let images from Database score higher (max->1), images by $G$ score lower (min->0).<br>

    Step 2: Fix discriminator $D$, and update generator $G$<br>

        Let images by $G$ score higher("fool" the discriminator $D$)

> In reality, we can use piece the two parts into together-so we only have "one" network. The first several layers for $G$. Rhe rest are $D$. Input a vector. Output a score $\in[0,1]$. A hidden-layer in the neural network outputs an image.

> If the machine knows "a man looking at left" and "a man looking at right", and there vectors are [-1,-1] and [1,1] respectively, then when we consider [0,0], its image is "a man looking in the middle"! We don't have to teach the machine Physics or even other things.

> Noise is not random filler. It provides the Generator with a *low-dimensional, continuous control space*.<br>

> We use log not just because outputs lie in $[0,1]$.<br>
> 1. Maximum Likelihood(Discriminator = binary classifier; log is the natural probabilistic choice.)<br>
> 2. Without log, confident predictions give near-zero gradients. <br>
Log *amplifies large mistakes*, keeping learning alive.
> 3. With optimal $D$, GAN minimizes  *Jensen–Shannon divergence* — which relies on the log form.
