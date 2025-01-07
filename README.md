## ANSyLF-DT

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

ANSyLF-DT (formerly NEXUS-DT) an advanced, integrated system designed for anomaly detection and adaptive control using a combination of neural networks, reinforcement learning, and symbolic reasoning. The project leverages CNN-LSTM models for detecting anomalies in sensor data, PPO agents for making adaptive control decisions, and Prolog-based symbolic reasoning for enhanced interpretability and decision transparency. Additionally, NEXUS-DT incorporates comprehensive evaluation and visualization tools to assess model performance and system behavior.

## Features

- **Anomaly Detection**: Utilizes CNN-LSTM models to identify anomalies in real-time sensor data.
- **Adaptive Control**: Implements PPO (Proximal Policy Optimization) agents to adaptively manage system parameters based on detected anomalies.
- **Symbolic Reasoning**: Integrates Prolog and ProbLog for rule-based reasoning, enhancing system interpretability and decision-making transparency.
- **Knowledge Graphs**: Generates and visualizes knowledge graphs representing system states, rules, and anomalies.
- **Comprehensive Evaluation**: Provides detailed metrics, classification reports, ROC/AUC curves, and Precision-Recall analyses.
- **Visualization Tools**: Offers a suite of visualization utilities for model architectures, feature importances, rule activations, and state transitions.
- **Robust Logging**: Implements extensive logging for monitoring system performance and debugging.

## Architecture

The ANSyLF-DT framework consists of modular layers that facilitate anomaly detection and adaptive control. Below is the framework architecture:

![ANSyLF-DT Code Architecture](src/media/images/ansylf_dt_code_architecture.png)

This diagram highlights key components such as the physical environment layer, processing layer, and adaptation layer, showing data flow and dynamic adaptation mechanisms.

The ANSyLF-DT system is modular, comprising the following key components:

1.  **Configuration Management**
    - Handles loading and validating YAML configuration files.

2.  **Core Functionality**
    - Integrates neural models, reinforcement learning agents, symbolic reasoning, and adaptive controllers.
    - Manages state updates, decision-making processes, and maintains historical data.

3.  **Reinforcement Learning**
    - Custom Gym environment tailored for NEXUS-DT.
    - Scripts for training PPO agents using Stable Baselines3.

4.  **Visualization**
    - Tools for visualizing model features, training metrics, rule activations, and knowledge graphs.

5.  **Symbolic Reasoning**
    - Prolog-based reasoning integrated with neural components.
    - ProbLog for probabilistic logic programming and uncertainty handling.

6.  **Evaluation**
    - Comprehensive evaluation metrics and visualization of model performance and rule effectiveness.

7.  **Utilities**
    - Model saving/loading, scaler management, and other helper functions.

8.  **Integration and Execution**
    - Orchestrates the training, evaluation, reasoning, and visualization processes.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Anaconda](https://www.anaconda.com/) (recommended for environment management)
- Prolog (e.g., [SWI-Prolog](https://www.swi-prolog.org/))
- ProbLog

### Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/sbhakim/ansylf-dt.git](https://github.com/sbhakim/ansylf-dt.git) 
    cd ansylf-dt
    ```

2.  **Set Up Python Environment**
    ```bash
    conda create -n ansylf_dt_env python=3.9 
    conda activate ansylf_dt_env
    ```

3.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Prolog and ProbLog**
    - SWI-Prolog: Follow the installation instructions from the official website.
    - ProbLog:
      ```bash
      pip install problog
      ```

5.  **Configure the Project**
    - Copy the example configuration file and adjust settings as needed:
      ```bash
      cp configs/config.yaml.example configs/config.yaml
      ```
    - Edit `configs/config.yaml` to set paths and parameters according to your environment.

## Usage

1.  **Train the CNN-LSTM Model**
    ```bash
    python src/training/trainer.py
    ```

2.  **Train the PPO Agent**
    ```bash
    python src/rl/train_ppo.py
    ```

3.  **Run the Main Pipeline**
    ```bash
    python main.py
    ```

4.  **Evaluate the Model**
    ```bash
    python src/evaluation/evaluation.py
    ```

5.  **Visualize Results**
    - Generated visualizations are saved in the `results/visualization/` directory.

## Project Structure

ansylf-dt/
├── configs/
│   └── config.yaml
├── logs/
│   └── nexus_dt.log
├── results/
│   ├── models/
│   ├── metrics/
│   ├── visualization/
│   │   ├── model_visualization/
│   │   ├── neurosymbolic/
│   │   └── knowledge_graphs/
│   └── knowledge_graphs/
├── src/
│   ├── config/
│   │   └── config_manager.py
│   ├── core/
│   │   ├── core.py
│   │   └── explainable.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_processing.py
│   ├── evaluation/
│   │   ├── evaluation.py
│   │   ├── pattern_metrics.py
│   │   └── init.py
│   ├── integration/
│   │   └── adaptive_controller.py
│   ├── logging/
│   │   └── logging_setup.py
│   ├── models/
│   │   └── cnn_lstm_model.py
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── reasoning/
│   │   ├── prob_rules.pl
│   │   ├── prob_query.py
│   │   ├── integrate_prob_log.pl
│   │   ├── rules.pl
│   │   ├── rule_learning.py
│   │   ├── knowledge_graph.py
│   │   ├── reasoning.py
│   │   └── init.py
│   ├── rl/
│   │   ├── nexus_dt_env.py
│   │   └── train_ppo.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── init.py
│   ├── utils/
│   │   ├── model_utils.py
│   │   └── init.py
│   └── visualization/
│       ├── model_visualization.py
│       ├── plotting.py
│       ├── neurosymbolic_visualizer.py
│       └── init.py
├── main.py
├── README.md
└── requirements.txt

## References

For more details and documentation on the tools and libraries used in this project, refer to the following resources:

- [ProbLog 2.2 Documentation](https://problog.readthedocs.io/en/latest/)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact Safayat at safayat.b.hakim@gmail.com.**