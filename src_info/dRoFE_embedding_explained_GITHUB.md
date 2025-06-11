# 🧠 Understanding `dRoFEEmbedding.py` – Line-by-Line Explanation

This document explains the Python code in `dRoFEEmbedding.py` by linking it directly to the theoretical equations from the XAIGUIFormer paper (ICLR 2025), with GitHub-compatible formatting.

---

## 📌 Mathematical Background

- **Equation (9)**: Defines rotary encoding based on frequency bounds (lower and upper).
- **Equation (10)**: Angular position per embedding dimension: `theta_t = (4π * t) / d`
- **Equation (11)**: Combines rotary encoding with demographic info:  
  `Re(R) = age × cos(f_l * theta_t) + gender`  
  `Im(R) = age × sin(f_u * theta_t) + gender`
- **Equation (12)**: Applies this rotation to Query/Key vectors:  
  `x_rot = R_complex * x = (Re(R) * x1 - Im(R) * x2) + i(Re(R) * x2 + Im(R) * x1)`

---

## 🧩 Code Explanation

### Initialization

```python
self.theta = 4 * math.pi * torch.arange(0, embedding_dim // 2) / embedding_dim
```

- Implements Equation (10): `theta_t = (4π * t) / d`
- Prepares angles used for rotation in each half-dimension.

---

### Frequency-bound Rotation

```python
f_l = freq_bounds[:, 0].unsqueeze(-1)
f_u = freq_bounds[:, 1].unsqueeze(-1)

angle_l = f_l * theta
angle_u = f_u * theta
```

- Implements Equation (9): generates rotation angles using frequency bounds.

---

### Demographic-Aware Rotation

```python
rot_real = age.unsqueeze(1) * torch.cos(angle_l) + gender.unsqueeze(1)
rot_imag = age.unsqueeze(1) * torch.sin(angle_u) + gender.unsqueeze(1)
```

- Implements Equation (11):
  - Combines cosine/sine components with age (scaling) and gender (bias).

---

### Embedding Rotation (Complex Multiplication)

```python
x1, x2 = x[..., ::2], x[..., 1::2]
x_rot_real = rot_real * x1 - rot_imag * x2
x_rot_imag = rot_real * x2 + rot_imag * x1
```

- Implements Equation (12): complex multiplication
  - Real part: `Re(R) * x1 - Im(R) * x2`
  - Imaginary part: `Re(R) * x2 + Im(R) * x1`

---

### Output Final Rotated Embedding

```python
x_rot = torch.stack([x_rot_real, x_rot_imag], dim=-1).flatten(-2)
```

- Recombines the rotated vectors into a real-valued embedding of size `[B, Freq, d]`.

---

## ✅ Summary Table

| Math Concept           | Code Line                            | Description                                     |
|------------------------|---------------------------------------|-------------------------------------------------|
| Frequency angle theta  | `self.theta = ...`                   | Equation (10)                                  |
| Band rotation angles   | `angle_l = f_l * theta`              | Equation (9)                                   |
| Rotary matrix R        | `rot_real`, `rot_imag`               | Equation (11)                                  |
| Complex multiplication | `x_rot_real`, `x_rot_imag`           | Equation (12)                                  |
| Final output           | `x_rot = ...`                        | Rotated embeddings used for Q/K                |

---

This module injects frequency-aware and demographic-aware inductive bias into token embeddings used as Query and Key inputs for attention mechanisms in Transformers.