# Adversarial Attack on Facial Recognition System with Predefined Spatial Constraints

From cancer diagnosis to self-driving cars, machine learning is profoundly changing the world. Recent studies show that state-of-the-art machine learning systems are vulnerable to adversarial examples resulting from small-magnitude perturbations added to the input. The broad use of machine learning systems makes it significant to understand the attacks on these systems where physical security and safety are at risk.

In this project, we focus on facial recognition systems, which are widely used in surveillance and access control. We develop and investigate resilient attacks that are physically realizable and inconspicuous, that allow an attacker to impersonate another individual. The investigation focuses on white-box attacks on the face-recognition systems. We develop an attack that will perturb only those facial regions that are normal to be changed for style, pose, fashion, etc. Our model automatically generates perturbations on a specific image for impersonation attack given the predefined spatial constraints. The attack evades the state-of-the-art face-recognition system with 100\% successful impersonation attack considering those spatial constraints. We compare and evaluate the efficacy of our method with state-of-the-art adversarial attacks that do not consider any constraints. Consequently, we propose some suggestions for the possible defenses for these types of spatially constrained attacks.

## Explored ML models:

- ResNet50
- Inception V3
- Convolutional Neural Network (CNN)

## Explored Datasets:

- MCS2018 dataset
- VGG_Face
- ImageNet

## Mask generation for spacial attack

![mask.png](images/mask.png)

## Project Team and Contributors

- [Sheik Murad Hassan Anik](https://www.linkedin.com/in/anik801/)
- Md Abdullah Al Maruf
