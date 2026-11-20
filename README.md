# QuantAttend

**A Comparative Study of Quantum and Classical Kernel SVMs for Classroom Face Detection**

QuantAttend classifies image patches from classroom videos as **Face** or **Background**, the first step towards automatic attendance. It compares a tuned classical RBF-SVM against a quantum fidelity-kernel SVM (simulated), using identical data and identical input features.

> **Research question:** Can a quantum feature map provide a more discriminative feature space than a classical RBF kernel for classroom face detection?

---

## Results

| Metric    | Classical RBF-SVM | Quantum Kernel SVM |
| --------- | ----------------- | ------------------ |
| Accuracy  | **95.56%**        | 77.04%             |
| Precision | **88.24%**        | 60.94%             |
| Recall    | **100%**          | 86.67%             |
| F1-score  | **93.75%**        | 71.56%             |

In this experiment the classical model performed better on every metric. This does **not** show that classical computing is always better than quantum computing; it only shows that this particular quantum setup did not beat a tuned classical model (see [Limitations](#limitations)).

---

## Pipeline

Both models share the same first stages:

1. **Frame extraction**: sample frames from classroom video at configured intervals.
2. **Face detection (MTCNN)**: detected faces become positive samples (label 1); background crops become negative samples (label 0).
3. **Feature extraction**: resize each crop to 32×32, convert to grayscale (1024 features).
4. **PCA**: reduce 1024 features to up to 4 components.

Then the models diverge:

### Classical approach
`StandardScaler → RBF-SVM (GridSearchCV)`

- Kernel: RBF, `K(x, z) = exp(-γ‖x − z‖²)`
- Hyperparameter search: `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, 0.01, 0.1, 1}`
- Trained on the full training set

### Quantum approach
`MinMaxScaler [0, π] → Quantum Feature Map → Fidelity Kernel → SVM (precomputed)`

- 4 qubits, one PCA feature per qubit
- Feature map: Hadamard → RY → RZ(2x) → CNOT ring (Q0→Q1→Q2→Q3→Q0)
- Kernel: `K(x, z) = |⟨φ(x)|φ(z)⟩|²`
- SVM: `SVC(kernel="precomputed", C=1.0, class_weight="balanced")`
- Training limited to `MAX_Q_TRAIN = 180` samples
- Executed with a **NumPy state-vector simulator** (no quantum hardware)

---

## Comparison

| Aspect                      | Classical                  | Quantum                          |
| --------------------------- | -------------------------- | -------------------------------- |
| Face detection              | MTCNN                      | MTCNN                            |
| Features                    | 32×32 grayscale → PCA (≤4) | 32×32 grayscale → PCA (≤4)       |
| Scaling                     | StandardScaler             | MinMaxScaler                     |
| Similarity measure          | RBF (distance-based)       | Quantum state fidelity           |
| Classifier                  | SVM                        | SVM with precomputed kernel      |
| Hyperparameter optimization | GridSearchCV               | Fixed C = 1.0                    |
| Training samples            | No explicit limit          | Max 180                          |
| Execution                   | Classical CPU              | NumPy state-vector simulation    |

---

## Getting Started

### Requirements

- Python 3.9+
- Jupyter Notebook / JupyterLab
- Libraries: `numpy`, `scikit-learn`, `opencv-python`, `mtcnn` (with its TensorFlow backend), `matplotlib`, `seaborn`

```bash
pip install numpy scikit-learn opencv-python mtcnn tensorflow matplotlib seaborn jupyter
```

### Run

1. Clone the repository and place your classroom video(s) in the folder the notebook expects.
2. Open the notebook:
   ```bash
   jupyter notebook
   ```
3. Run all cells in order. The notebook extracts frames, builds the dataset, trains both models, and prints the comparison metrics.

> Adjust the notebook name, video paths, and configuration values (frame interval, `SEED`, `MAX_Q_TRAIN`) to match your setup.

### Inspect model errors

Add a cell after the comparison section to plot confusion matrices for both models and see whether errors are false positives (background called face) or false negatives (missed faces).

---

## Limitations

- **Simulated quantum circuit.** Results come from a classical simulator; no quantum hardware or quantum advantage is demonstrated.
- **Pseudo-labels.** Labels are generated automatically by MTCNN and background cropping, not manually verified. Reported accuracy measures agreement with these labels, not true attendance-count accuracy.
- **Uneven tuning.** The classical model is tuned with GridSearchCV; the quantum model uses a fixed `C` and a fixed feature map with no trainable parameters.
- **Limited quantum training data.** At most 180 randomly selected samples are used, which may also be class-imbalanced.
- **Small circuit.** Only 4 qubits and a simple feature map are used.
- **Not a trainable quantum model.** This is a quantum kernel method (fixed circuit + classical SVM), not a variational quantum classifier.

---

## Future Work

- Try a more expressive feature map (e.g., ZZ feature map) and different entanglement patterns.
- Tune the quantum SVM (`C`) and add a bandwidth/scaling parameter, using cross-validation.
- Use balanced sampling for the quantum training subset.
- Replace raw pixels with pretrained CNN embeddings (e.g., MobileNetV2, EfficientNet) reduced by PCA, shared by both models.
- Use manually verified labels and group-based splits by video or frame.
- Repeat over multiple random seeds and report mean ± standard deviation.
- Evaluate against independently verified face counts for real attendance accuracy.
- Test on real quantum hardware or noisy simulators.

---

## Project Structure

```
QuantAttend/
├── notebook.ipynb     # Full pipeline: data, classical model, quantum model, comparison
├── videos/            # Input classroom videos
└── README.md
```

*(Update to match your actual repository layout.)*

---

## License

Add your preferred license here (e.g., MIT).

## Acknowledgements

- [MTCNN](https://github.com/ipazc/mtcnn) for face detection
- [scikit-learn](https://scikit-learn.org/) for PCA, SVM, and model selection
