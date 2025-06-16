# Understanding `trainer.py` – End-to-End Forward Pipeline

This document explains the full architecture implemented in `trainer.py`, including how the different modules (dRoFE, Transformer, XAI, and loss) interact. It also includes a full input/output trace and model-level reasoning.

---

## Overall Pipeline Structure

The `Trainer` class coordinates the entire XAIGuiFormer architecture:

1. Prepares Q/K using `dRoFE` to inject EEG frequency and demographic information.
2. Passes Q to a **Vanilla Transformer** to generate coarse attention representations.
3. Applies an XAI **explainer** (e.g. DeepLIFT) to extract token importance maps.
4. Uses these maps as Qexpl/Kexpl in a **XAI-Guided Transformer** to refine attention.
5. Produces two outputs: `logits_coarse`, `logits_refined`.
6. If labels are provided, computes a **joint supervised loss**.

---

## Step-by-Step Architecture

```
Trainer/
│
├── Inputs:
│   ├── x_raw        : Tensor [B, Freq, d]      # Token embeddings from Connectome Tokenizer
│   ├── freq_bounds  : Tensor [Freq, 2]         # EEG band bounds [f_l, f_u]
│   ├── age          : Tensor [B, 1]            # Patient age
│   ├── gender       : Tensor [B, 1]            # Patient gender
│   └── y_true       : Tensor [B] or None       # Ground truth labels
│
├── Step 1 – dRoFE Encoding:
│   ├── q = W_q(x_raw), k = W_k(x_raw)
│   ├── q_rot = dRoFE(q, freq_bounds, age, gender)
│   └── k_rot = dRoFE(k, freq_bounds, age, gender)
│
├── Step 2 – Vanilla Transformer:
│   ├── x_coarse = VanillaTransformer(q_rot)
│   └── logits_coarse = Classifier(mean(x_coarse))     # [B, C]
│
├── Step 3 – XAI Explainer (TODO):
│   ├── q_expl = DeepLIFT(x_coarse)
│   └── k_expl = DeepLIFT(x_coarse)
│   ⚠️ Currently mocked as q_expl = q_rot.detach(), k_expl = k_rot.detach()
│
├── Step 4 – XAI-Guided Transformer:
│   ├── x_refined = XAIGuidedTransformer(x_raw, q_expl, k_expl)
│   └── logits_refined = Classifier(mean(x_refined))   # [B, C]
│
├── Step 5 – Joint Loss:
│   └── total_loss = (1 - alpha) * CE(logits_coarse, y_true) + alpha * CE(logits_refined, y_true)
│
└── Outputs:
    ├── If y_true is None:
    │     └── logits_coarse: [B, C]
    │     └── logits_refined: [B, C]
    └── Else:
          └── total_loss: scalar
```

---

## Key Design Insights

- Demographic encoding (dRoFE) allows the model to personalize frequency token embeddings.
- XAI guidance enforces **meaningful attention** based on saliency maps.
- The joint loss encourages **alignment** between coarse and refined predictions.

---

## Remaining TODOs

- Integrate actual `ConnectomeTokenizer` output into `x_raw`
- Use a real `explainer.py` implementation (e.g., Captum’s DeepLIFT)

---

This class is central to orchestrating the reproducibility of XAIGuiFormer from preprocessing to prediction and XAI-guided refinement.