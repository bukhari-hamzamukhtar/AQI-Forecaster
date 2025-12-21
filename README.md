# 🌍 End-to-End Serverless MLOps for AQI Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/bukhari-hamzamukhtar/aqi-forecaster)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/docker-automated-blue)](https://www.docker.com/)

[cite_start]An end-to-end MLOps system that predicts the **Air Quality Index (AQI)** for the next 72 hours based on the past 7 days of pollutant data[cite: 10]. [cite_start]The system features an automated pipeline for data ingestion, feature engineering, model training, and drift monitoring[cite: 7].

**🚀 Live Demo:** [Click here to view the deployed App on Hugging Face](https://huggingface.co/spaces/bukhari-hamzamukhtar/aqi-forecaster)

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Performance](#-performance)
- [How to Run Locally](#-how-to-run-locally)
    - [Method 1: Docker (Recommended)](#method-1-docker-recommended)
    - [Method 2: Manual Python Setup](#method-2-manual-python-setup)
- [Tech Stack](#-tech-stack)

---

## 🔭 Project Overview
Air pollution is a silent killer. [cite_start]This project aims to provide accurate, short-term AQI forecasting to help citizens make informed health decisions[cite: 5]. 

[cite_start]The system fetches hourly pollutant data ($PM_{2.5}$, $PM_{10}$, $NO_2$, $SO_2$, $O_3$, $CO$) from the **Open-Meteo API**[cite: 27], processes it through a robust feature engineering pipeline, and generates forecasts using an ensemble of machine learning models. [cite_start]It was designed to handle **data drift** and operates in a serverless environment using Docker[cite: 8].

## 🏗 System Architecture
The pipeline consists of three main stages:
1. [cite_start]**Data Pipeline:** Automated ingestion and Feature Store (Local Parquet fallback implemented after Hopsworks)[cite: 30, 78].
2. [cite_start]**Training Pipeline:** Comparative evaluation of XGBoost, Random Forest, SVM, MLP, and SARIMA[cite: 7].
3. [cite_start]**Inference & Monitoring:** Streamlit dashboard with SHAP explainability and KS-test drift detection[cite: 9, 20].

## ✨ Key Features
* [cite_start]**Multi-Model Evaluation:** compares 5 different architectures (SARIMA proved best for local constraints)[cite: 132].
* [cite_start]**Lazy Model Loading:** Fetches large model artifacts (~8GB) from Hugging Face Hub on runtime to keep Docker images light[cite: 8, 80].
* [cite_start]**Explainable AI:** Integrated **SHAP** and **LIME** to explain *why* a specific AQI was predicted[cite: 9].
* [cite_start]**Drift Detection:** Monitors production data distributions using the Kolmogorov-Smirnov test to flag when the model needs retraining[cite: 20].
* [cite_start]**Reproducibility:** Fully containerized using Docker[cite: 8].

## 📊 Performance
[cite_start]The system was evaluated in two phases: Cloud (Hopsworks) and Local (Resource-Constrained)[cite: 9].
* [cite_start]**Best Model (Local):** SARIMA [cite: 132]
* [cite_start]**RMSE:** ~24.07 [cite: 180]
* [cite_start]**Accuracy:** >90% on test set validation[cite: 74].

---

## 💻 How to Run Locally

Since this project deals with large model files and specific dependencies, the easiest way to run it is via Docker.

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
* A **Hugging Face Token** (Read permission) to download the model. [Get one here](https://huggingface.co/settings/tokens).

### Method 1: Docker (Recommended)
[cite_start]This ensures you run the exact same environment as the production deployment[cite: 104].

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bukhari-hamzamukhtar/AQI-Forecaster.git](https://github.com/bukhari-hamzamukhtar/AQI-Forecaster.git)
    cd AQI-Forecaster
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t aqi-forecaster .
    ```

3.  **Run the container:**
    *Replace `YOUR_HF_TOKEN` with your actual Hugging Face token.*
    ```bash
    docker run -p 8501:8501 -e HF_TOKEN="YOUR_HF_TOKEN" aqi-forecaster
    ```

4.  **Access the App:**
    Open your browser and go to `http://localhost:8501`.

### Method 2: Manual Python Setup
If you want to edit the code or run without Docker.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:**
    [cite_start]You need to set the `HF_TOKEN` variable in your terminal so the app can download the model[cite: 112].
    ```bash
    # Windows (PowerShell)
    $env:HF_TOKEN="your_token_here"
    
    # Mac/Linux
    export HF_TOKEN="your_token_here"
    ```

4.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```

---

## 🛠 Tech Stack
* [cite_start]**Core:** Python 3.11[cite: 107], Pandas, NumPy
* [cite_start]**ML Frameworks:** Scikit-learn, TensorFlow/Keras, XGBoost, Statsmodels (SARIMA) [cite: 7]
* [cite_start]**MLOps:** Docker, GitHub Actions, Hugging Face Hub (Model Registry) [cite: 8]
* [cite_start]**App Interface:** Streamlit [cite: 81]

## 📝 Cite This Project
If you use this project in your research, please cite:
```bibtex
@software{aqi_forecaster_2025,
  author = {Bukhari, Hamza Mukhtar},
  title = {End-to-End Serverless MLOps System for AQI Prediction},
  year = {2025},
  url = {[https://github.com/bukhari-hamzamukhtar/AQI-Forecaster](https://github.com/bukhari-hamzamukhtar/AQI-Forecaster)}
}

## 📚 References
1. **WHO**, "WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide," WHO, Geneva, 2021.
2. **L. Breiman**, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.
3. **T. Chen and C. Guestrin**, "XGBoost: A scalable tree boosting system," KDD '16, pp. 785-794, 2016.
4. **S. M. Lundberg and S.-L. Lee**, "A unified approach to interpreting model predictions," NIPS, vol. 30, 2017.
5. **M. T. Ribeiro, S. Singh, and C. Guestrin**, "Why should I trust you?: Explaining the predictions of any classifier," KDD '16, pp. 1135-1144, 2016.
6. **D. Sculley et al.**, "Hidden technical debt in machine learning systems," NIPS, vol. 28, 2015.
7. **D. Merkel**, "Docker: lightweight Linux containers for consistent development and deployment," Linux Journal, 2014.
8. **D. Kreuzberger et al.**, "Machine learning operations (MLOps): Overview, definition, and architecture," IEEE Access, vol. 11, 2023.
9. **F. Pedregosa et al.**, "Scikit-learn: Machine learning in Python," JMLR, vol. 12, 2011.
10. **U.S. EPA**, "Technical Assistance Document for the Reporting of Daily Air Quality," EPA 454/B-18-007, 2018.

## 📜 License
This project is licensed under the MIT License.

## 🤝 Acknowledgments
* [cite_start]**Open-Meteo** for the excellent free Air Quality API[cite: 7].
* [cite_start]**GIKI Faculty** for guidance on MLOps best practices[cite: 269].
