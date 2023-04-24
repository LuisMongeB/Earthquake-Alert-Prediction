
<div align="center">
<h1 align="center">
<br>
Earthquake Alert Prediction :volcano:
</h1>
</div>


---
## 📍 Table of Contents
- [📍 Table of Contents](#-table-of-contents)
- [👋 Introdcution](#-introdcution)
- [⚙️ Project Structure](#️-project-structure)
- [🏎💨 Getting Started](#-getting-started)
  - [✅ Prerequisites](#-prerequisites)
  - [💻 Usage](#-installation)
- [📫 Contact](#-contact)

---

## 👋 Introdcution

This project is aimed at predicting earthquake alerts, given a few variables as independent variables like depth, significance, comunity decimal intensities (CDI) and others. The dataset source is: https://www.kaggle.com/datasets/warcoder/earthquake-dataset


---

## ⚙️ Project Structure

```bash
.
├── LICENSE
├── Makefile
├── README.md
├── conf
│   ├── badges.json
│   ├── conf.toml
│   ├── file_extensions_map.toml
│   └── language_instructions.toml
├── docs
│   ├── README_EX_1.md
│   ├── README_EX_2.md
│   ├── README_EX_3.md
│   ├── imgs
│   │   ├── docs.png
│   │   ├── header.png
│   │   ├── misc.png
│   │   ├── setup.png
│   │   ├── toc.png
│   │   └── tree.png
│   └── raw_data.csv
├── requirements.txt
├── scripts
│   ├── clean.sh
│   ├── run.sh
│   └── test.sh
├── setup
│   ├── environment.yaml
│   └── setup.sh
├── setup.py
├── src
│   ├── __init__.py
│   ├── builder.py
│   ├── conf.py
│   ├── logger.py
│   ├── main.py
│   ├── model.py
│   ├── processor.py
│   └── utils.py
└── tests
    ├── conftest.py
    ├── test_builder.py
    ├── test_conf.py
    ├── test_logger.py
    ├── test_main.py
    ├── test_model.py
    ├── test_processor.py
    └── test_utils.py

8 directories, 40 files
```
---

<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-src-open.svg" width="80" />

## 🏎💨 Getting Started

### ✅ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> python 3.11.0
> anaconda package manager

### 💻 Installation

1. Clone the repository:
```sh
git clone https://github.com/LuisMongeB/earthquake-alert-prediction
```

2. Change to the project directory:
```sh
cd earthquake-alert-prediction
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### Usage

```sh
python train.py
```

<hr />


## 📫 Contact

If you have any questions or concerns, feel free to contact me through LinkedIn `[https://www.linkedin.com/in/luis-diego-monge-bolanos/]`

---
