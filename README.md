# Robust Multiclass Human Activity Recognition

An image-based **15-class Human Activity Recognition (HAR)** system built with CNNs and transfer learning , with a focus not just on **accuracy**, but on **robustness and failure analysis**.

### What was explored?

**CNN Baseline -> Transfer Learning -> Explainability -> OOD Testing -> Robustness Improvement**

| Stage               | Model / Approach             |                Result |
| ------------------- | ---------------------------- | --------------------: |
|   Baseline         | Scratch CNN                  |            **43.85%** |
|   Improved         | EfficientNet-V2-S            |            **84.97%** |
|   Hierarchical     | Coarse → Fine classification |            **84.30%** |
|   Robust Training | Corruption-aware fine-tuning | **80.50% Wild Score** |
|   TTA               | Test-Time Augmentation       |            **85.41%** |

### What makes it different?

Instead of stopping at validation accuracy, the model was **stress-tested under distribution shifts**:

`Gaussian Noise` · `Blur` · `Motion Blur` · `Low Light` · `Low Contrast` · `JPEG` · `Occlusion` · `Hue Shift`

The model's **Wild/OOD score improved from 79.22% -> 80.50%**, with motion-blur performance improving by nearly **6 percentage points**.

### Explainability & Failure Analysis

**Grad-CAM** was used to investigate what the CNN focuses on, while corruption-level analysis was used to identify failure modes.

One of the strongest failure patterns was:

> **Texting + Motion Blur**

This helped turn the project from a simple classifier into a study of **when and why HAR models fail**.

### Project Notebooks

**01 - Baseline HAR**
EDA, preprocessing, dataset splitting, DataLoaders and scratch CNN.

**02 - Robust Multiclass HAR**
EfficientNet-V2-S transfer learning, strong augmentation, Grad-CAM, hierarchical classification and OOD evaluation.

**03 - Improved Robustness**
Corruption-aware fine-tuning, failure analysis, TTA and advanced robustness experiments.

### Core Idea

> **Don't just ask whether the model is accurate. Ask whether it still works when the world changes.**

**Tech:** Python · PyTorch · CNN · EfficientNet-V2-S · Transfer Learning · Grad-CAM · OOD Evaluation · Data Augmentation
