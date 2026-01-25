# 🧩 Preprocessing & Tiling

This document describes the **preprocessing and tiling strategy** used in the Rooflytics project.  
These steps transform raw high-resolution aerial imagery into a **memory-efficient, illumination-robust dataset** suitable for deep learning–based roof segmentation.

---

## 1️⃣ Why Preprocessing and Tiling Are Required

The aerial imagery used in this project consists of **very high-resolution orthophotos (~10,000 × 10,000 pixels)**. Directly training a neural network on such images is impractical due to:

- GPU out-of-memory (OOM) issues  
- Extremely slow training and inference  
- Limited batch sizes  
- Large background regions compared to roof pixels  

Preprocessing and tiling ensure that the data is:
- GPU-friendly  
- Numerically stable  
- Illumination-consistent  
- Suitable for efficient batch training  

---

## 2️⃣ Image Tiling

### Motivation

Deep learning segmentation models such as U-Net require fixed-size inputs and benefit from localized spatial context. Large aerial images must therefore be split into smaller patches.

### Tiling Strategy

Each aerial RGB image and its corresponding binary roof mask are divided into fixed-size tiles:

