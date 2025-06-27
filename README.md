# Gossip World Models: Stabilizing Generative Dynamics through Social Consistency

**Project Status:** In Development

This repository contains the PyTorch implementation for the "Gossip World Models" research project. We propose a novel regularization technique to improve the long-term stability and physical plausibility of generative world models.

[//]: # (Add a link to your paper or a longer-form explanation here if you have one)
<!-- [Read the Paper (Link)]() | [Project Page (Link)]() -->

<p align="center">
  <img src="https://i.imgur.com/your-diagram-placeholder.png" width="700" alt="Diagram illustrating the gossip protocol">
  <em>Figure 1: A conceptual diagram of the Gossip Protocol. Two independent models (A and B) dream about the future. They then "gossip" by showing each other the final frame of their dream. A consistency loss penalizes a model if it finds the other's dream "inconceivable," forcing both to learn a shared, plausible version of reality.</em>
</p>

## 1. The Idea

A key failure mode of generative world models is **reality drift**, where long-term simulations (dreams) diverge into physically impossible scenarios. An agent can exploit these flaws, learning policies that fail in the real environment.

Our solution, the **"Gossip" Protocol**, is a procedural check inspired by social consensus.
1.  **Two independent world models** (`A` and `B`) are trained simultaneously.
2.  Both models "dream" a future trajectory from the same starting point.
3.  They "show" each other the final frame of their dream.
4.  A **consistency loss** penalizes a model if it cannot comprehend the other's dream.
    `Loss_A = || z_A_N - Encoder_A(Decoder_B(z_B_N)) ||²`
This "social pressure" forces the models to discard idiosyncratic fantasies and converge on the essential, shared dynamics of the environment.