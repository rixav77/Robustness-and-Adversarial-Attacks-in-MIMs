# Robustness and Adversarial attacks in MIMs

## **Introduction: Robustness and Adversarial Attacks in Masked Image Modeling (MIM)**

Ever since Meta’s 2021 paper ‘Masked auto-encoders are scalable self supervised learners’ came into existence, Masked Image Modeling (MIM) has become famous as a key strategy for self supervised learning. Validated by the success of masked language models like BERT, MIM models like MAEs, SimMIM and BEiT promise to train powerful models for vision problems, with better downstream performance than even supervised pretraining on the ImageNet-1K dataset.

![Screenshot 2025-04-14 at 3.03.53 PM.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/Screenshot_2025-04-14_at_3.03.53_PM.png)

The proposal is straightforward:

> Mask some patches. Learn from the visible patches. Predict the missing patches, while learning good representations in the process.
> 

## **But what happens when these models are fooled?**

Though widely accepted for learning feature representations, MIMs see images through statistical glasses, quite contrary to how humans process visuals. This allows some room for technical foulplay, where a minute change in the input, say changing some pixel values, makes the model to give a completely different, mostly irrelevant output. Such a model is said to be under an **ADVERSARIAL ATTACK.** 

This leads us to an important question:

> Are MIM-based models robust against adversarial attacks??
> 

This is what we explore under **"Robustness and Adversarial Attacks in MIM"** - how well MIM models can handle tricky inputs, and what kinds of mathematical or algorithmic methods can help improve their reliability.

## **What Will This Work Cover?**

We will explore this topic step by step:

1. **How exactly MIM works?**
2. **What are Adversarial Attacks?**
3. **Robustness in MIM**
4. **Evaluating MIM under Attacks**
5. **Improving Robustness**
6. **Code Snippets and Visualizations**

## **1. Introduction to Masked Image Modeling (MIM)**

Before going on to formal definitions, let us focus on what we are trying to achieve. Though an integral part of the MIM training process, image generation is hardly the end goal we look for while constructing an MIM architecture. 

What matters to us is what the model learns in the process. When we mask some parts of an image, our purpose at its very core is that the model now shifts all of its focus to what it has: learning from the visible patches. In this way, while providing minimal information, we are creating a powerful model that knows how to gather meaningful features in harder pretraining environments.

So in the architecture we typically have a setup of gathering latent features from the input image and organising them into a latent space (an encoder) and another setup which would assess the encoders feature extraction ability.

This idea can be approached in different MIM methods. At large, there are two major types:

1. Reconstruction-Based MIM
2. Contrastive-Based MIM

**RECONSTRUCTIVE- BASED MIM**

We take an image, and break it down into patches, adding positional encoding into each patch. After this, we mask some of the patches randomly. The visible patches are passed into an encoder, which tries to extract meaningful features from the input and then map them into a latent space. These latent representations, along with the position embeddings of the missing patches is provided to the decoder, which based on these inputs try to predict the missing patches

![146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png)

The output of the decoder is matched with the input and a square-errored loss function is used for training.

$$
L_{reconstruction}=∑_{i∈masked}∥D(E(x_{visible}))_i−x_i∥^2, 
$$

where  $x_{visible}:$ 

- The visible (unmasked) part of the input

**$E(x_{visible}):$**

- This is the **encoder output**.

**$D(E(x_{visible})):$**

- The **decoder output**.

**$D(E(x_{visible}))_i:$**

- The **reconstructed value** at position iii, where iii belongs to the **masked indices**.

**$x_i:$**

- The **true/original value** at the same position iii (from the original input x).

Some of the famous models that use this approach:

- **MAE (Masked Autoencoders)**
- **SimMIM**
- **PeCo**

**CONTRASTIVE-BASED MIM**

Think of a big playground. In contrastive learning, your model would pull the features corresponding to the same output together in this playground, while pushing away the ones corresponding to a different output. Hence, for let us say a trivial case of a binary classification where you are trying to classify if you can pet an animal or not, it would pull the cat and dog features together and push the banana features away.

![ChatGPT Image Apr 16, 2025, 11_23_25 AM.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/ChatGPT_Image_Apr_16_2025_11_23_25_AM.png)

Applying this into contrastive-based MIMs, the model would learn semantic similarity between masked and unmasked views.

$$
Let:x ∈ ℝ^{H×W×C}
$$

denote an input image. When this image is partitioned into a set of non-overlapping patches {x₁, x₂, ..., xₙ}, each patch has dimensions P×P×C. A binary mask vector m ∈ {0,1}ᴺ determines which patches are visible (mᵢ = 1) and which are masked (mᵢ = 0).

The MIM objective is defined as learning a function $f_θ$ (typically implemented as a transformer encoder-decoder architecture) that reconstructs the masked patches:

![Screenshot 2025-04-15 at 4.09.53 PM.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/Screenshot_2025-04-15_at_4.09.53_PM.png)

The loss function is formulated as a reconstruction loss between the predicted and true masked patches:

$$
L_{MIM} = 1/|M| ∑_{i∈M}||x̂_i - x_i||²
$$

where M = {i : mi = 0} is the set of masked indices.

## **2. What are Adversarial Attacks?**

Adversarial attacks take benefit of the vulnerabilities of a model. They spot loopholes and try to change the model’s output by introducing subtle changes in the input. In the context of **Masked Image Modeling (MIM)**, these attacks target the reconstruction ability of the model, not classification - making the attack objective subtly different but potentially more damaging in pretraining scenarios.

---

### **Mathematical Formulation**

Let 

$$
\mathbf{x} \in \mathbb{R}^{H \times W \times C}
$$

 be a clean input image and

$$
 \hat{\mathbf{x}} = f_\theta(\mathbf{x}_{\text{visible}})
$$

 be the reconstructed output for masked tokens. An adversarial perturbation 

$$
δ∈R^{H×W×C}
$$

 is crafted such that: $∥δ∥_p≤ϵ$  for a small $ϵ>0$ bounded in $L_p$ normand the perturbed image  $x′=x+δ$ yields a significantly worse reconstruction.

Formally, an adversary seeks:

$$
δ^∗=arg⁡ max⁡_{∥δ∥p≤ϵ }L_{MIM}(f_θ((x+δ)_{visible}),x_{masked})
$$

Even though x’ appears visually identical to x the model may produce $\hat{{x}}$’ that is structurally inconsistent or semantically incoherent.

---

### **Why Do These Attacks Work?**

Several theoretical and empirical insights explain this fragility:

- **Local Linearity Hypothesis**: First, one important phenomenon to keep in mind is how neural networks behave in high-dimensional spaces. Surprisingly, although these models are built to handle complex patterns, they often act in ways that are almost linear when you zoom in locally. This **local linearity** means that small changes in the input, if aligned with the right direction, can lead to surprisingly large shifts in the output. That’s a fundamental reason why certain adversarial attacks can be so effective.
- **High Curvature of Loss Landscape**: Second, let’s consider the shape of the loss landscape. Especially in large, overparameterized models, the loss surface isn’t smooth and flat—it has sharp ridges and valleys. This **high curvature** means that a very tiny nudge in the input can send the gradients soaring, leading to big jumps in loss or unexpected behavior during optimization.
- **Overfitting to Masking Patterns**: And finally, there's the issue of how MIMs deal with missing information. Because they’re trained to work with specific kinds of masked inputs, they sometimes become too specialized-learning to excel at reconstructing from a particular masking pattern but failing to generalize beyond it. This **overfitting to masking patterns** can cause the model to lose awareness of the global structure of an image when faced with unfamiliar kinds of occlusions.

---

### **Attack Strategies in Vision**

The most common attack algorithms include:

- **Fast Gradient Sign Method (FGSM)**: One of the simplest and most well-known methods is the **Fast Gradient Sign Method (FGSM)**. The idea here is pretty straightforward: we take the gradient of the loss with respect to the input image and nudge the image slightly in the direction that increases the loss the most. Mathematically, this looks like:

$$
x′=x+ϵ⋅sign(∇_xL)
$$

      Here, ϵ controls how strong the perturbation is.

### 🎯 FGSM Attack Implementation

```python
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)
```

- **Projected Gradient Descent (PGD)**:  Building on this, we have **Projected Gradient Descent (PGD)** - essentially FGSM done in steps. After each small perturbation, the input is projected back into a constrained region (an ϵ-ball) to make sure the attack stays within bounds.
- **AutoAttack / CW / BIM**: Beyond these, there are more advanced attack methods like **AutoAttack**, **Carlini & Wagner (CW)**, and **Basic Iterative Method (BIM)**. These go even deeper - trying to get around defenses like gradient masking and taking advantage of the complex, non-convex nature of the model’s loss landscape.

Now, what’s interesting in the case of MIM is that we’re not just looking at whether the model predicts the right class — we care about **how well it reconstructs the image**. So instead of accuracy, we evaluate the attacks using measures like **PSNR** (Peak Signal-to-Noise Ratio), **SSIM** (Structural Similarity Index), or **LPIPS** (Learned Perceptual Image Patch Similarity), which tell us how close the output is to the original image in terms of both structure and perceptual quality.

---

![Screenshot 2025-04-16 at 11.59.39 AM.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/Screenshot_2025-04-16_at_11.59.39_AM.png)

- A clean vs. adversarial input example with visual similarity.

![ChatGPT Image Apr 16, 2025, 12_00_18 PM.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/ChatGPT_Image_Apr_16_2025_12_00_18_PM.png)

- The resulting reconstructed outputs under a MIM like MAE — showing how subtle changes drastically affect reconstructions.

## **3. Robustness in Masked Image Modeling (MIM)**

Masked Image Modeling wasn’t originally built with robustness in mind. Its goal was to enable self-supervised learning - helping models understand images without needing labels, by simply predicting the parts of an image that are hidden. But what’s fascinating is how this **pretext task** -the seemingly simple job of filling in missing patches - ends up changing how these models respond to adversarial noise.

Unlike traditional vision models that are trained to assign a single label to an entire image, MIMs learn to focus on structure, context, and fine-grained detail - because that’s what they need to reconstruct the image from what little they’re given. And it turns out that this different way of looking at an image also changes how the model behaves when we introduce perturbations.

So, to truly understand MIMs, we need to explore this **robustness landscape** - how they react when the input is tweaked or attacked, and what makes their behavior both surprisingly resilient in some ways and oddly fragile in others. It’s this interplay between their training objective and their learned representations that makes the story of MIM robustness so intriguing.

---

### **Is MIM More Robust Than Supervised Pretraining?**

Some studies (e.g., *DeiT, MAE vs ViT under attack*) suggest that MIM-pretrained models can **transfer better robustness** to downstream tasks than their supervised counterparts. Why might that be?

Intuitively:

- **MIM learns to recover missing information**, so it relies more on **global context** than just local pixel-level cues.
- Adversarial perturbations usually **attack local sensitivities** — which might not fool a model that depends on broader context.

But this robustness is **not consistent**, and in many cases, MIM models can **fail miserably** under attacks that target **masked token recovery** directly.

---

### **A Geometric Perspective**

Imagine input images as points on a **data manifold** $M⊂R^d$ 

The goal of a robust model is to ensure that small perturbations δ do not push x+δ **off-manifold** — i.e., into a region where the model has never been trained.

Masked pretraining naturally encourages the model to learn **distributional consistency** over the visible tokens, implicitly modeling the structure of M. But:

- MIM **does not model the joint distribution p(x)** directly — it models **conditional distributions** like $p(x_{masked}∣x_{visible}$)
- So when $x_{visible}$ is adversarially corrupted, the conditional distribution can shift sharply, **breaking the reconstruction**.

---

### **Why MIMs Fail Under Adversarial Perturbations**

Let’s express the loss again:

$$
\mathcal{L}_{\text{MIM}} = \mathbb{E}_{\mathcal{M}} \left[\ell\left(f_\theta(\mathbf{x}_{\text{visible}}), \mathbf{x}_{\text{masked}}\right)\right]
$$

If $x_{visible}$ lies off-manifold due to adversarial perturbation, the model might decode:

- Textures that are **not coherent** with global semantics,
- Or hallucinated patterns that minimize the loss but are **semantically incorrect**.

This also explains why **MIMs trained on low-resolution datasets** (e.g., CIFAR-10) are often more brittle than those trained on high-resolution, large-scale datasets like ImageNet.

---

### **Empirical Observations**

When we put MIM models to the test against adversarial attacks, some interesting patterns start to emerge.

For one, models like **MAE** and **SimMIM** tend to hold up better than their supervised counterparts - especially when there's plenty of training data. In these high-data regimes, they show more resistance to attacks like PGD, suggesting that the way they learn representations gives them a certain edge. But that advantage quickly fades in low-data scenarios, where they start to crack just like any other model.

As we ramp up the intensity of attacks - say, using PGD with more steps - all MIMs eventually start to break down. What’s curious, though, is *how* they break. Their **reconstruction errors don’t just increase - they explode in different ways**, depending on the model and the nature of the attack. It’s not a uniform collapse.

![download (10).png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/download_(10).png)

And perhaps the most striking observation: even if we only corrupt the **visible patches** — the ones the model actually sees — the damage ripples out. Because the model relies so heavily on these for predicting the masked regions, a small corruption can act like a **bad seed**, triggering a cascade of errors throughout the reconstructed image. It’s a fragile balance — and one that shows how tightly MIMs are wired to their inputs.

![Screenshot 2025-04-16 at 12.18.27 PM.png](Robustness%20and%20Adversarial%20attacks%20in%20MIMs%201d6875b1759880b0930ffd5bafc40ec7/Screenshot_2025-04-16_at_12.18.27_PM.png)

---

## **5. Improving Robustness in Masked Image Modeling**

Masked Image Models operate under a simple assumption: that the **visible patches are clean** — untouched, reliable, and representative of the image. The model learns to reconstruct what’s missing by focusing on what’s present. But what happens when that very foundation is shaken? What if the visible patches themselves are subtly, maliciously perturbed?

This section walks through emerging strategies — both **algorithmic and architectural** — that aim to make MIMs more robust in the face of adversarial threats.

### Adversarial Training (Simplified)

Even this simple joint training helps make the model’s representations smoother and reconstructions more stable.

```python
for image in dataloader:
image = image.cuda()
image.requires_grad = True
output = model(image)
loss = loss_fn(output, ground_truth)
loss.backward()
perturbed = fgsm_attack(image, epsilon, image.grad.data)
output_adv = model(perturbed)
loss_adv = loss_fn(output_adv, ground_truth)

total_loss = loss + loss_adv
total_loss.backward()
optimizer.step()

```

---

### **1. Adversarial Training for MIMs**

At its core, adversarial training asks the model to prepare for the worst. Instead of only learning from clean images, the model is now exposed to **intentionally perturbed inputs** — ones crafted to maximize its reconstruction error.

The objective becomes a game of two players:

$$
min_{θ} 
  
E
_{x∼D}

[
_{∥δ∥≤ϵ}
max

L
_{MIM}

(x+δ)]
$$

- **Inner maximization**: Find a perturbation  $δ$ that *maximally* disrupts the model’s predictions.
- **Outer minimization**: Update the model parameters θ so that it can *withstand* even the worst-case input.

### Intuition:

You’re nudging the model toward **Lipschitz continuity** — where small changes in input produce only small changes in output. That kind of stability is exactly what we want when the real world throws noise and uncertainty our way.

---

### **2. Robust Masking Strategies**

Random masking - like what we see in standard MAEs - treats all patches equally. But not all patches carry the same weight. Some hold critical structure, while others are just background noise.

To build robustness, we can **bias the masking process** during training:

- **Saliency-aware masking**: Target important regions (e.g. edges, objects) more often. This exposes the model to tougher reconstruction challenges.
- **Entropy-guided masking**: Mask unpredictable or high-variance patches more frequently, encouraging the model to develop strong priors about image structure.

In both cases, the idea is to **stress-test** the model during training, so that it generalizes better when the input is less than perfect.

---

### **3. Denoising Objectives**

If we expect the model to face noisy inputs, why not explicitly teach it to denoise?

We can modify the loss function to include a **denoising term**:

$$
 Loss\mathcal{L} = \underbrace{\| f_\theta(\mathbf{x}_{\text{visible}}) - \mathbf{x}_{\text{masked}} \|_2^2}_{\text{Reconstruction Loss}} + \lambda \underbrace{\| f_\theta(\mathbf{x}_{\text{adv}}) - \mathbf{x} \|_2^2}_{\text{Denoising Loss}}
$$

The decoder is now trained not just to reconstruct from clean inputs, but to **map adversarial examples back to the clean version**. This teaches the model to “look past the noise.”

---

### **4. Architectural Modifications**

The architecture itself can be tweaked to absorb shocks from noisy data.

### a. **Noise-Resistant Encoders**

Introduce elements like:

- **Stochastic depth**: Randomly drop layers during training.
- **Dropout**: Randomly drop units within layers.
- **Spectral normalization**: Control the Lipschitz constant of layers.

Each of these forces the model to **learn smoother, more distributed representations** - less likely to break under stress.

### b. **Ensemble Encoding**

Why rely on a single path? Use **parallel encoders** - one processing clean inputs, the other handling perturbed ones - and **combine their features**. This “ensemble of perspectives” helps the model anchor itself even when the input is noisy.

---

### **5. Regularization via Contrastive Objectives**

Finally, contrastive learning offers a neat way to **align representations** from clean and adversarial views of the same image.

$$
{L}_{\text{contrastive}} = - \log \frac{\exp(\text{sim}(z, z'))}{\sum_{i} \exp(\text{sim}(z, z_i))}
$$

Here, z and z′ are latent vectors from clean and adversarial inputs. The loss encourages these to be **close in the embedding space**, while pushing away unrelated examples zi.

This means the model isn’t just memorizing what to reconstruct — it’s learning *how* to represent things in a way that’s **invariant to small, adversarial changes**.

---

All of these strategies push in the same direction: they aim to **break the assumption that visible patches are always clean**. Whether through smarter training, better objectives, or architectural flexibility, they teach the model not just to reconstruct - but to **reason and adapt**, even when the input world gets a little messy.

---

## **Conclusion**

Masked Image Modeling (MIM) has become one of the most exciting directions in computer vision, allowing models to learn by simply predicting missing parts of an image - no labels needed. But while these models like MAE and SimMIM are great at capturing visual structures, there’s an elephant in the room: **they can be fooled pretty easily**.

We’ve seen how **tiny, almost invisible changes** to the visible parts of an image - called **adversarial perturbations** - can throw these models off. They start predicting the wrong thing for the masked patches, even though the original image barely changed to our eyes. That’s a big problem if we’re planning to use these models in real-world tasks like medical imaging, self-driving, or security systems.

We explored why this happens: the masked parts depend heavily on the unmasked ones, so if the visible patches are slightly corrupted, the whole reconstruction process falls apart. We also looked at how the reconstruction error increases as we make the perturbations stronger - even though the perturbations remain visually unnoticeable.

The good news? This vulnerability also opens up new research directions:

- Can we design **more robust masking strategies**?
- Should we **train MIMs with adversarial examples** from the start?
- What if we **add denoising or smoothing modules** to clean up corrupted inputs?

As MIMs become more powerful and widespread, we’ll need to ensure they aren’t just accurate - but also **reliable under pressure**.
