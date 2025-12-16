# 🤖Deep Learning and Reinforcement Learning Assignment

| Category          | Information                                         |
| ----------------- | --------------------------------------------------- |
| **Name**          | Mahin Khanum M                                      |
| **USN**           | 1CD22AI035                                          |
| **Semester**      | 7th                                                 |
| **Department**    | Artificial Intelligence and Machine Learning (AIML) |
| **Subject**       | Deep Learning and Reinforcement Learning            |
| **Course Code**   | BAI701                                              |
| **Academic Year** | 2025–2026                                           |

This repository contains a curated collection of **cleaned, modernized, and beginner-friendly implementations** of key **Deep Learning (DL)** and **Reinforcement Learning (RL)** concepts.

The primary goal of this assignment is to **study classic models and algorithms**, identify limitations in basic implementations, and enhance them using **modern best practices** while keeping the code **readable, reproducible, and reusable**.

### 🎯 Focus Areas

* Correct use of modern TensorFlow/Keras APIs
* Proper tensor shapes and data pipelines
* Reproducibility using fixed random seeds
* Modular, readable, and reusable code
* Meaningful improvements without unnecessary complexity

---

## 📋 Contents

1. **AlexNet (Deep Learning – CNN)**
2. **Deep Reinforcement Learning (Q-Learning on Graph)**
3. **LSTM (Time-Series Forecasting)**
4. **RNN (Character-Level Language Model)**
5. **Tic-Tac-Toe Reinforcement Learning (Self-Play)**

---

# 1️⃣ AlexNet

### ✅ Original Implementation

The original AlexNet implementation included:

1. Basic AlexNet architecture using Conv2D layers
2. Manual configuration for ImageNet (1000 output classes)
3. No Batch Normalization layers
4. No reusable class structure or training utilities

While functional, the implementation lacked **modern regularization, stability improvements, and flexibility**.

---

### ✅ Improvements Made

**File:** `alexnet.ipynb`

A **modernized AlexNet-style CNN** implemented using `tf.keras.Sequential`, following current deep learning best practices.

### 🔧 Key Enhancements

* **Configurable First Convolution Stride**

  * Parameter: `first_conv_stride` (default = `2`)
  * Can be set to `4` to replicate original AlexNet downsampling

* **Batch Normalization After Every Convolution**

  * Improves training stability and convergence speed

* **Separate Activation Layers**

  * `BatchNormalization()` → `Activation('relu')`
  * Cleaner and more maintainable architecture

* **He Initialization & L2 Regularization**

  * `kernel_initializer='he_normal'`
  * `kernel_regularizer=l2(weight_decay)` (default `5e-4`)

* **Removed Redundant Bias Terms**

  * `use_bias=False` in convolution layers followed by BatchNorm

* **Pooling with `padding="same"`**

  * Prevents unintended spatial dimension collapse

* **Reusable Class-Based API**

  ```python
  AlexNet(
      input_shape=(224, 224, 3),
      num_classes=1000,
      first_conv_stride=2,
      weight_decay=5e-4
  )
  ```

### ▶ How to Run

```bash
python alexnet_modern.py
```

---

# 2️⃣ Deep Reinforcement Learning

### ✅ Original Implementation

1. Traditional Q-learning using a Q-matrix
2. Simple graph-based navigation problem
3. Basic visualization without reproducibility

The original version had **logical bugs**, relied on **global variables**, and used **legacy NumPy constructs**.

---

### ✅ Improvements Made

**File:** `deep reinforcement learning.ipynb`

### 🔧 Key Enhancements

* **Bug Fix**

  * Removed incorrect global variable usage during action sampling

* **Modern NumPy Usage**

  * Replaced `np.matrix` with standard NumPy arrays

* **Reproducible Graph Layout**

  * `networkx.spring_layout(seed=...)` ensures consistent visualization

* **Clear Modular Structure**

  * Functions are broken into:

    * `build_graph`
    * `build_reward_matrix`
    * `available_actions`
    * `sample_next_action`
    * `train_q`
    * `extract_path`
    * `update_with_environment`

* **Environment-Aware Action Biasing**

  * Tracks simulated “police” and “drug” traces
  * Influences agent behavior dynamically

* **Improved Safety & UX**

  * Fixed random seeds
  * Fallback logic for edge cases
  * Informative prints and plots

### 📦 Requirements

```bash
pip install numpy matplotlib networkx
```

### ▶ How to Run

```bash
python improved_q_learning_graph.py
```

---

# 3️⃣ LSTM (Time-Series Forecasting)

**File:** `lstm.ipynb`

### ✅ Original Implementation

1. Single LSTM layer with 10 units
2. Basic sequence prediction
3. Hardcoded Windows file paths
4. Minimal visualization and error handling

---

### ✅ Key Improvements Made

* Consistent use of `tensorflow.keras`
* Correct LSTM input shape:

  ```
  (samples, timesteps, features)
  ```
* Modular helper functions:

  * `safe_load_series()`
  * `create_sequences()`
* CSV fallback:

  * Generates synthetic time-series data if file is missing
* Reproducible random seeds
* EarlyStopping callback to prevent overfitting
* Minimal and portable dependencies

### ▶ How to Run

With CSV:

```bash
python simple_lstm_passengers.py path/to/data.csv
```

Without CSV:

```bash
python simple_lstm_passengers.py
```

### 📦 Dependencies

```bash
pip install tensorflow pandas matplotlib scikit-learn numpy
```

---

# 4️⃣ RNN (Character-Level Language Model)

**File:** `rnn.ipynb`

A character-level RNN that predicts the next character in a sequence.

### ✅ Original Implementation

1. `SimpleRNN` layer with 50 units
2. Small text corpus
3. Fixed sequence length
4. Greedy decoding using `argmax`

---

### ✅ Improvements Made

* Uses `tensorflow.keras` consistently
* Clear data preprocessing pipeline using `to_categorical`
* Modular helper functions:

  * `build_vocab`
  * `create_dataset`
  * `build_model`
  * `generate_text`
* **Temperature-based sampling**

  * Controls creativity vs determinism
* Robust handling of generation seeds
* Reproducible training behavior

### ▶ How to Run

```bash
python simple_char_rnn.py --epochs 80 --seed "The handsome "
```

### 📦 Dependencies

```bash
pip install tensorflow numpy
```

---

# 5️⃣ Tic-Tac-Toe Reinforcement Learning (Self-Play)

**File:** `tic_tac_toe_rl.ipynb`

A reinforcement learning demo where two agents learn Tic-Tac-Toe through **self-play**, after which a human can play against the trained agent.

### ✅ Original Implementation

1. Q-learning with state-value dictionary
2. Simple exploration strategy
3. Basic text-based gameplay

---

### ✅ Key Fixes & Improvements Made

* Correct initial player symbol logic
* Robust CLI argument parsing (works in notebooks)
* Safe policy save/load using `pathlib`
* Reduced default training rounds for faster experimentation
* Clear training progress output
* Cleaner class structure and type hints

### ▶ How to Run

```bash
python tic_tac_toe_rl.py
```

Optional arguments:

```bash
python tic_tac_toe_rl.py --rounds 10000 --verbose_every 2000
```

---

## 🎯 Conclusion

This assignment demonstrates how **classic Deep Learning and Reinforcement Learning examples** can be significantly improved by applying **modern coding practices and theoretical correctness**.

### ✅ Key Takeaways

* Cleaner APIs reduce bugs and improve maintainability
* Reproducibility is essential for experimentation
* Modular code is easier to understand and extend
* Small improvements can greatly enhance learning value

The repository serves as a **solid reference** for understanding, experimenting with, and extending fundamental DL and RL models.

