
<div align="center">
<h1 align="center">
<br>
:volcano: Earthquake Alert Prediction :volcano:
</h1>
</div>


---
## 📍 Table of Contents
- [📍 Table of Contents](#-table-of-contents)
- [👋 Introduction](#-introduction)
- [🏎💨 Getting Started](#-getting-started)
  - [✅ Prerequisites](#-prerequisites)
  - [💻 Usage](#-usage)
- [📫 Contact](#-contact)

---

## 👋 Introduction

This project is aimed at predicting earthquake alerts, given a few independent variables such as depth, significance, comunity decimal intensities (CDI) and others.

The dataset source is: https://www.kaggle.com/datasets/warcoder/earthquake-dataset

Main challenge was to address class imbalance, which was tackled with by synthetically upsampling the minority class (SMOTE). 

After hyperparameter tuning, the best model was the RandomForestClassifier with an F1 score of: 96.9%. 

---

## 🏎💨 Getting Started

### ✅ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> python version 3.11.0

### 💻 Installation

1. Clone the repository:
```sh
git clone https://github.com/LuisMongeB/earthquake-alert-prediction
```

2. Change to the project directory:
```sh
cd earthquake-alert-prediction
```

3. Create and activate conda environment:
```sh
conda create -n earthquakes python=3.11.0 && conda activate earthquakes
```

4. Install the dependencies:
```sh
pip install -r requirements.txt
```

### Usage

```sh
python main.py
```

<hr />


## 📫 Contact

If you have any questions or concerns, feel free to contact me through [LinkedIn](https://www.linkedin.com/in/luis-diego-monge-bolanos/).

---
