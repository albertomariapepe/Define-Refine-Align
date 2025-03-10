# Define, Refine, Align (DRA)

## Overview 📌
Define, Refine, Align (DRA) is a streamlined pipeline for aligning unordered 3D line sets **without the need for prior correspondence evaluation**. By leveraging **Geometric Algebra** \( G(4,0,0) \), DRA efficiently estimates relative poses through hypercomplex networks, ensuring robust and real-time 3D line alignment.

Unlike conventional methods that require costly correspondence estimation and iterative optimization, DRA directly predicts the alignment transformation, making it well-suited for **real-time applications** where **noisy data acquisition** and **fast processing** are crucial.

## Methodology 🧩
DRA consists of three core modules:

1. **Define** (Feature Extraction):
   - Uses an **attention-based feature extractor** to identify putative matches between source and target line bundles.
   - Extracts meaningful representations from unordered 3D line sets.

2. **Refine** (Pose Estimation):
   - Includes an **Equivariant Module** \(\phi\) and a **Rotational Module** \(\rho\) operating in \( G(4,0,0) \).
   - Maps 3D lines to their respective poses using geometric transformations.

3. **Align** (Pose Averaging & Normalization):
   - Aggregates estimated poses \(M_{\rho,1}, M_{\rho,2}, M_{\phi,1}, M_{\phi,2}\).
   - Normalizes the final transformation \( \hat{M} \) to align source and target line bundles.

### Mathematical Formulation ✏️
Given a source line bundle $\mathscr{L}_S$ and target line bundle$ \mathscr{L}_T $, DRA estimates the transformation $M$ that aligns them:

$ \mathscr{L}_T = M \mathscr{L}_S \tilde{M} $

This direct estimation bypasses the need for explicit correspondence evaluation, significantly reducing computational overhead.

## Performance 📊
Compared to the state-of-the-art correspondence-free model, PlückerNet Regression, DRA reduces pose estimation errors by 30% to 90%, showcasing its superior performance in various scenarios. This makes it a highly promising approach for real-time applications that require fast and accurate alignment without relying on costly correspondence estimation.

## Installation & Usage ⚙️
Our model is implemented in **PyTorch**.
Clone the repository and install dependencies:
```bash
 git clone https://github.com/albertomariapepe/Define-Refine-Align
 cd DRA
 pip install -r requirements.txt
```
Train:
```bash
python main_train.py
```

Test:
```bash
python main_test.py
```

## Contacts 💡 
ap2219 [at] cam [dot] ac [dot] uk
We believe geometry-informed pipelines like DRA can pave the way for robust and efficient 3D line registration in the future.
