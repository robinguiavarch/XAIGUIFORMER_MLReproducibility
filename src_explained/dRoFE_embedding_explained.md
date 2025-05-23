# 🧠 Understanding `dRoFEEmbedding.py` – Line-by-Line Explanation 

This document explains the Python code in `dRoFEEmbedding.py` by linking it directly to the theoretical equations from the XAIGUIFormer paper (ICLR 2025).

---

## 📌 Mathematical Background

- **Equation (9)**: Defines rotary encoding based on frequency bounds (lower and upper).
- **Equation (10)**: Angular position per embedding dimension: $ \theta_t = \frac{4\pi t}{d} $
- **Equation (11)**: Combines rotary encoding with demographic info (age × rotation + gender bias).
- **Equation (12)**: Applies this rotation to Query/Key vectors in the attention mechanism.

---

## 🧩 Code Explanation

### Initialization

```python
self.theta = 4 * math.pi * torch.arange(0, embedding_dim // 2) / embedding_dim
```

- Implements Equation (10): $ \theta_t = \frac{4\pi t}{d} $
- Prepares angles used for rotation in each half-dimension.

---

### Frequency-bound Rotation

```python
f_l = freq_bounds[:, 0].unsqueeze(-1)  # lower bounds
f_u = freq_bounds[:, 1].unsqueeze(-1)  # upper bounds

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
    - Multiplies cosine/sine components by age (amplitude),
    - Adds gender as bias.

---

### Embedding Rotation (Complex Multiplication)

```python
x1, x2 = x[..., ::2], x[..., 1::2]
x_rot_real = rot_real * x1 - rot_imag * x2
x_rot_imag = rot_real * x2 + rot_imag * x1
```

- Equivalent to complex multiplication:
    - $ \text{Re}(R) \cdot x_1 - \text{Im}(R) \cdot x_2 $
    - $ \text{Re}(R) \cdot x_2 + \text{Im}(R) \cdot x_1 $

---

### Output Final Rotated Embedding

```python
x_rot = torch.stack([x_rot_real, x_rot_imag], dim=-1).flatten(-2)
```

- Recombines rotated parts to get $ x'_f = R^d_{f,Demog} \cdot x_f $
- Matches Equation (12).

---

## ✅ Summary Table

| Math Element | Equation | PyTorch Code |
|--------------|----------|--------------|
| Angles \(\theta_t\) | (10) | `self.theta = ...` |
| Frequency Phases \(f_l, f_u\) | (9) | `angle_l = f_l * theta` |
| Demographic Rotation | (11) | `rot_real`, `rot_imag` |
| Complex Product | (12) | `x_rot_real`, `x_rot_imag` |
| Output Token | — | `x_rot = ...` |

---

This module effectively injects **frequency-specific** and **subject-specific (age/gender)** inductive bias into the Query and Key vectors used in Transformer attention.