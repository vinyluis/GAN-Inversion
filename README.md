# Unsupervised GANs
Experiments done with Unsupervised GANs for my Master's Degree dissertation.

## Objective
The objective of the experiment is to create generators capable of generating a latent vector that can be used for image manipulation.

## Experiment Page
The complete experiment can be accessed on (PT-BR only): 

https://www.notion.so/vinitrevisan/N-o-Supervisionada-cc97dd35ef954aed8023bfe0b265853b

---

## Contents

***main.py***

File with the experiment core. Training, test and validation of the networks. Parameters of the experiment.

***networks.py***

File with the classes and functions to create the generators and discriminators used on the experiments.

***losses.py***

File with the functions used to evaluate the losses to train the GANs

***utils.py***

File with all utilities functions, such as plot control, image processing, and exception handling.

***transferlearning.py***

File with the functions used to train networks using the Transfer Learning approach.

***validate.py***

Tests vector interpolation to see how the reconstruction of interpolated images is working for a given generator.

---

## Also part of this project

Autoencoders ([github](https://github.com/vinyluis/Autoencoders))

Pix2Pix-CycleGAN ([github](https://github.com/vinyluis/Pix2Pix-CycleGAN))

Unsupervised-GANs ([github](https://github.com/vinyluis/Unsupervised-GANs))

Main experiment page ([Notion [PT-BR]](https://www.notion.so/vinitrevisan/Estudos-e-Experimentos-4027d5e5256e4efc80fe1cd4dc018553))
