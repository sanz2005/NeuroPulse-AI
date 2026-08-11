🧠 NeuroPulse AI

Neuromorphic Healthcare Intelligence Platform for Multi-Modal Biosignal Monitoring and Anomaly Detection Using Spiking Neural Networks

<p align="center">
  <strong>A software-simulated neuromorphic healthcare intelligence platform for ECG, EEG, and EMG analysis using Spiking Neural Networks (SNNs), explainable AI, real-time signal monitoring, benchmarking, and energy-aware analysis.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Spiking%20Neural%20Networks-SNN-purple" alt="SNN">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Next.js-Frontend-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/TypeScript-Frontend-3178C6?logo=typescript" alt="TypeScript">
  <img src="https://img.shields.io/badge/Research-Prototype-orange" alt="Research Prototype">
</p>

📌 Overview

NeuroPulse AI is an end-to-end neuromorphic healthcare intelligence platform designed to investigate how Spiking Neural Networks (SNNs) can be applied to heterogeneous biomedical time-series signals.

The platform brings together three biosignal modalities:

Modality

Signal

Primary Focus

❤️ ECG

Electrocardiogram

Cardiac anomaly / arrhythmia analysis

🧠 EEG

Electroencephalogram

Seizure / neurological anomaly analysis

💪 EMG

Electromyogram

Muscle activity / anomaly analysis

Instead of treating biomedical signals only as conventional continuous-valued time series, NeuroPulse AI converts signals into event-based spike representations and processes them using spiking neuron models.

End-to-End Pipeline

Biosignal Processing
        ↓
Spike Encoding
        ↓
SNN Inference
        ↓
Explainability
        ↓
Benchmarking
        ↓
Visualization
        ↓
Healthcare Monitoring

The system is implemented as a software-based neuromorphic simulation platform, making it possible to study neuromorphic intelligence without requiring dedicated neuromorphic hardware.

🎯 Research Motivation

Conventional deep-learning approaches such as CNNs, LSTMs, and Transformers can provide strong performance on biomedical signals, but they generally operate on dense numerical representations.

Neuromorphic computing introduces a different computational paradigm:

⚡ Event-driven processing

🕸️ Sparse spike activity

⏱️ Temporal information processing

🧠 Biologically inspired neuron dynamics

🔋 Potential reductions in computational activity and energy consumption

However, biomedical SNN research is often presented as isolated experiments for a single modality or task.

NeuroPulse AI addresses this gap by integrating multiple biosignal modalities into one research-oriented platform with model benchmarking, explainability, spike visualization, and energy/complexity analysis.

✨ Key Features

🫀 1. Multi-Modal Biosignal Intelligence

The platform is structured around three independent signal pipelines:

Modality

Signal

Primary Task

ECG

Electrocardiogram

Cardiac anomaly / arrhythmia analysis

EEG

Electroencephalogram

Seizure / neurological anomaly analysis

EMG

Electromyogram

Muscle activity / anomaly analysis

Each modality has dedicated preprocessing and training components.

⚡ 2. Spike-Based Signal Encoding

Continuous biomedical signals are transformed into spike trains before SNN processing.

Implemented Encoding Approaches

📈 Rate Encoding

⏱️ Temporal Encoding

⚡ Delta Modulation

This enables the system to represent signal changes as temporal events rather than relying exclusively on dense continuous-valued input.

🧠 3. Spiking Neural Network Models

The SNN layer contains reusable neuron and surrogate-gradient components.

Core Implementations

🧠 Leaky Integrate-and-Fire (LIF) neuron

🎯 Surrogate-gradient based learning

❤️ ECG SNN

🧠 EEG SNN

💪 EMG SNN

The modular design makes it possible to extend the system with additional neuron models and spike-processing strategies.

📊 4. Conventional Deep Learning Baselines

To evaluate the usefulness of the neuromorphic approach, NeuroPulse AI includes conventional baselines:

🧩 CNN

🔁 LSTM

🤖 Transformer

The project therefore supports comparative analysis rather than evaluating the SNN in isolation.

🔬 5. Explainable AI for Spiking Models

The platform includes SNN-oriented explainability components for interpreting model decisions.

The XAI pipeline produces visual artifacts such as:

🧠 Spike attribution maps

🔥 Heatmaps

📊 Modality-specific attribution visualizations

The frontend exposes these insights through dedicated visualization panels.

⚙️ 6. Energy and Computational Analysis

NeuroPulse AI includes research utilities for:

📊 Model benchmarking

🧮 Computational complexity analysis

🔋 Energy comparison

🕸️ Spike sparsity analysis

📈 Training-history visualization

⚖️ Performance comparison

The goal is not only to ask:

Which model is more accurate?

but also:

How much computation and spike activity is required to achieve that performance?

📡 7. Real-Time Healthcare Monitoring Architecture

The backend provides API routes for:

📡 Signal analysis

🧠 Inference

👤 Patient records

🚨 Alerts

📊 Benchmarking

🔍 Explainability

📈 Signal handling

WebSocket infrastructure is also included for real-time communication and streaming-oriented functionality.

🖥️ 8. Interactive Research Dashboard

The Next.js frontend provides dedicated interfaces for:

📈 Biosignal analysis

🧪 Research / benchmarking

🧠 Neural-twin style visualization

🩺 Clinical reasoning

🚨 Clinical priority information

🕒 Decision timelines

👨‍⚕️ Doctor-oriented summaries

🔍 XAI visualization

🕸️ Spike raster visualization

🔋 Energy monitoring

The dashboard is designed to make complex SNN behavior understandable through visual analytics rather than exposing only raw prediction outputs.

🏗️ System Architecture

                         ┌──────────────────────────┐
                         │       NeuroPulse AI      │
                         │ Neuromorphic Healthcare  │
                         │      Intelligence        │
                         └────────────┬─────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
        ┌─────────┐              ┌─────────┐              ┌─────────┐
        │   ECG   │              │   EEG   │              │   EMG   │
        └────┬────┘              └────┬────┘              └────┬────┘
             │                        │                        │
             ▼                        ▼                        ▼
        Preprocessing           Preprocessing           Preprocessing
             │                        │                        │
             └───────────────┬────────┴────────┬───────────────┘
                             ▼
                    Spike Encoding Layer
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
             Rate         Temporal      Delta Modulation
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                     SNN Processing Layer
                  ┌──────────┼──────────┐
                  │          │          │
                  ▼          ▼          ▼
                ECG SNN    EEG SNN    EMG SNN
                  │          │          │
                  └──────────┼──────────┘
                             ▼
                 ┌────────────────────────┐
                 │ Prediction / Anomaly   │
                 │ Detection / Inference  │
                 └───────────┬────────────┘
                             │
          ┌──────────────────┼───────────────────┐
          ▼                  ▼                   ▼
     Explainability     Benchmarking        Observability
          │                  │                   │
          ▼                  ▼                   ▼
     Attribution       CNN/LSTM/Transformer   Spike Activity
       Heatmaps          Comparison           Sparsity
                                             Energy/Complexity
                             │
                             ▼
                  ┌──────────────────────┐
                  │   FastAPI Backend    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Next.js / TypeScript │
                  │ Research Dashboard   │
                  └──────────────────────┘

📁 Project Structure

NeuroPulse AI/
│
├── ai/
│   ├── encoding/
│   │   ├── delta_modulation.py
│   │   ├── rate_encoding.py
│   │   └── temporal_encoding.py
│   │
│   ├── evaluation/
│   │   ├── advanced_evaluation.py
│   │   ├── benchmarking.py
│   │   ├── complexity_analysis.py
│   │   ├── generate_ecg_benchmark.py
│   │   └── plot_training_history.py
│   │
│   ├── models/
│   │   ├── baselines/
│   │   │   ├── cnn_model.py
│   │   │   ├── lstm_model.py
│   │   │   └── transformer_model.py
│   │   │
│   │   └── snn/
│   │       ├── ecg_snn.py
│   │       ├── eeg_snn.py
│   │       ├── emg_snn.py
│   │       ├── lif_neuron.py
│   │       └── surrogate.py
│   │
│   ├── preprocessing/
│   │   ├── ecg_processor.py
│   │   ├── eeg_processor.py
│   │   └── emg_processor.py
│   │
│   ├── training/
│   │   ├── train_ecg.py
│   │   ├── train_ecg_transformer.py
│   │   ├── train_eeg.py
│   │   ├── train_emg.py
│   │   └── trainer.py
│   │
│   └── xai/
│       └── spike_attribution.py
│
├── backend/
│   └── app/
│       ├── api/
│       │   ├── routes/
│       │   │   ├── alerts.py
│       │   │   ├── analyze.py
│       │   │   ├── benchmark.py
│       │   │   ├── inference.py
│       │   │   ├── patients.py
│       │   │   ├── records.py
│       │   │   ├── signals.py
│       │   │   └── xai.py
│       │   │
│       │   └── websocket/
│       │       └── manager.py
│       │
│       ├── config.py
│       ├── database.py
│       └── main.py
│
├── frontend/
│   ├── app/
│   │   ├── analyze/
│   │   ├── neural-twin/
│   │   ├── research/
│   │   └── page.tsx
│   │
│   ├── components/
│   │   ├── clinical/
│   │   ├── ui/
│   │   └── visualization/
│   │
│   ├── lib/
│   │   ├── api.ts
│   │   └── websocket.ts
│   │
│   └── store/
│       └── analysisStore.ts
│
├── notebooks/
│   └── visualize_data.py
│
├── outputs/
│   ├── plots/
│   ├── tables/
│   └── xai/
│
├── check_results.py
├── .gitignore
└── README.md

Note: Large raw datasets, processed arrays, virtual environments, local databases, and model binaries are intentionally excluded from GitHub through .gitignore.

🧬 Data Sources

The research pipeline is designed around publicly available biomedical datasets.

ECG — MIT-BIH Arrhythmia Database

Used for ECG signal processing and cardiac anomaly / arrhythmia experiments.

EEG — CHB-MIT Scalp EEG Database

Used for seizure-oriented EEG analysis.

EMG — NinaPro

Used for EMG-based muscle activity analysis and anomaly-oriented experiments.

Dataset Policy

The repository does not store the large raw/processed datasets.

This keeps the Git repository lightweight and avoids unnecessarily redistributing large biomedical datasets.

Download the datasets from their respective official/public sources and place them under the expected local data/ structure before running preprocessing or training.

🔄 End-to-End Workflow

1. Data Acquisition

Biomedical signals are obtained from public datasets.

ECG → MIT-BIH
EEG → CHB-MIT
EMG → NinaPro

2. Signal Preprocessing

Each modality has a dedicated processor responsible for preparing the signal for model input.

Raw Signal
    ↓
Cleaning / Filtering
    ↓
Normalization / Standardization
    ↓
Segmentation
    ↓
Model-Ready Signal

3. Spike Encoding

The processed signal is transformed into an event-based representation.

Continuous Signal
       ↓
Spike Encoder
       ↓
Spike Train
       ↓
SNN

4. SNN Inference

Spike trains are processed through LIF-based SNN architectures using surrogate-gradient learning.

5. Baseline Comparison

The same research pipeline can be compared against:

SNN
CNN
LSTM
Transformer

6. Explainability

The SNN output is accompanied by attribution / heatmap visualizations to investigate which temporal signal regions contributed to model decisions.

7. Evaluation

The evaluation layer generates:

📊 Benchmark summaries

📈 Training histories

🧮 Complexity analysis

🔋 Energy comparisons

🕸️ Spike sparsity analysis

🧩 Confusion matrices

📈 ROC curves

📊 Precision-recall curves

🎯 Calibration plots

🕸️ Radar comparisons

8. Visualization

The generated research artifacts are exposed through the frontend dashboard for interactive analysis.

📈 Evaluation & Research Outputs

The repository contains generated research artifacts under outputs/.

Visualization Categories

outputs/
├── plots/
│   ├── biosignal overview
│   ├── dataset distribution
│   ├── spike train visualization
│   ├── dataset summary
│   ├── accuracy comparison
│   ├── energy comparison
│   ├── spike sparsity
│   ├── benchmark summary
│   ├── training histories
│   ├── ROC curves
│   ├── PR curves
│   ├── calibration
│   ├── radar comparison
│   ├── complexity analysis
│   └── confusion matrices
│
├── tables/
│   └── benchmark / complexity results
│
└── xai/
    ├── ECG SNN attribution
    ├── ECG SNN heatmap
    ├── EEG SNN attribution
    ├── EEG SNN heatmap
    ├── EMG SNN attribution
    └── EMG SNN heatmap

These outputs are intended to support both model-performance evaluation and neuromorphic behavior analysis.

🧪 Model Benchmarking

One of the core research components is the comparison between neuromorphic and conventional architectures.

Baseline Models

Convolutional Neural Network

Long Short-Term Memory Network

🤖 Transformer

Neuromorphic Model

Spiking Neural Network with LIF neurons

The evaluation framework is designed to compare models across multiple dimensions:

Dimension

Purpose

Predictive performance

Evaluate classification / anomaly detection quality

Training behavior

Compare convergence and learning curves

Computational complexity

Analyze model / resource requirements

Spike sparsity

Quantify event-driven activity

Energy-oriented metrics

Study potential efficiency advantages

Explainability

Understand model decisions

Note: Exact experimental metrics should be reported from the generated benchmark files rather than hard-coded into this README.

🔍 Explainable AI

A major component of NeuroPulse AI is making SNN predictions interpretable.

XAI Pipeline

Signal
  ↓
Spike Encoding
  ↓
SNN
  ↓
Prediction
  ↓
Spike Attribution
  ↓
Heatmap / Temporal Explanation

This allows researchers to inspect which portions of a biosignal were associated with the model's decision.

The frontend includes dedicated XAI components for presenting these explanations.

⚡ Why Spiking Neural Networks?

Traditional neural networks process dense numerical activations.

SNNs communicate information using discrete spikes and temporal dynamics.

Traditional Neural Network

Input → Dense Activations → Dense Computation → Output

Spiking Neural Network

Input → Spike Encoding → Sparse Events → Neuron Dynamics → Output

Potential advantages investigated by this project include:

⚡ Event-driven computation

⏱️ Temporal information representation

🕸️ Sparse neural activity

🧠 Neuromorphic compatibility

🔋 Energy-aware computation

📡 Suitability for future edge / real-time biomedical systems

NeuroPulse AI focuses on empirically studying these properties, rather than assuming that SNNs are automatically more efficient.

🛠️ Technology Stack

Layer

Technologies

AI / ML

Python, PyTorch, NumPy, SciPy, Scikit-learn

Neuromorphic AI

SNNs, LIF neurons, surrogate-gradient learning

Backend

FastAPI, Python, WebSocket, database layer

Frontend

Next.js, React, TypeScript, CSS

Data

MIT-BIH, CHB-MIT, NinaPro

Research Outputs

NumPy, JSON, CSV, PNG

Development

Git, GitHub, VS Code

🚀 Installation

Prerequisites

Recommended environment:

Python 3.10+
Node.js 18+
npm
Git

Clone the Repository

git clone https://github.com/sanz2005/NeuroPulse-AI.git
cd NeuroPulse-AI

🐍 Backend / AI Environment

Create a Python virtual environment:

Windows

py -3.10 -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1

Install AI dependencies:

pip install -r ai/requirements.txt

If the backend in your local working version has its own dependency file, install those dependencies as well.

🖥️ Frontend Setup

Move into the frontend:

cd frontend

Install dependencies:

npm install

Run the development server:

npm run dev

Then open the local development URL shown by Next.js.

🔌 Backend Setup

From the project root, activate the Python environment and start the FastAPI application using the project's configured entry point.

Typical development command:

uvicorn backend.app.main:app --reload

If your local configuration uses a different module path or port, use the configuration defined in backend/app/.

🧪 Running the AI Pipeline

The AI code is organized into separate stages.

Preprocessing

ai/preprocessing/

Contains modality-specific processing for:

ECG

EEG

EMG

Training

ai/training/

Includes training scripts for:

ECG

EEG

EMG

ECG Transformer

Evaluation

ai/evaluation/

Includes:

📊 Benchmarking

Advanced evaluation

🧮 Complexity analysis

Training-history plotting

Benchmark generation

Explainability

ai/xai/

Contains spike attribution functionality.

📊 Research Dashboard

The frontend is organized around several research and clinical-analysis views.

Analysis

/analyze

Provides signal-analysis and AI reasoning functionality.

Research

/research

Provides research-oriented visualization and benchmarking functionality.

Neural Twin

/neural-twin

Provides a neuromorphic / visual analytics-oriented interface for interpreting system behavior.

🧩 API Modules

The backend is organized into modular API routes:

/api/routes/
├── alerts
├── analyze
├── benchmark
├── inference
├── patients
├── records
├── signals
└── xai

This separation allows model inference, clinical information, benchmarking, alerts, and explainability to evolve independently.

🔐 Data & Privacy

NeuroPulse AI is a research prototype and should not be treated as a certified medical device.

Important considerations:

📚 Public datasets are used for research.

🗂️ Large datasets are intentionally excluded from the Git repository.

🗄️ Local databases are excluded.

🔐 Environment files and secrets are excluded through .gitignore.

⚠️ Model outputs should be interpreted as research results, not clinical diagnoses.

⚠️ Current Scope

The current implementation is focused on:

🧠 Software-based neuromorphic simulation

🧬 Multi-modal biosignal analysis

⚡ SNN experimentation

⚖️ Baseline comparison

🔍 Explainability

🔋 Energy / complexity analysis

🖥️ Interactive visualization

It does not claim hardware-level neuromorphic execution.

Future deployment on specialized neuromorphic hardware can be explored as an extension.

🔮 Future Work

🧠 Advanced Neuromorphic Models

SNN-Transformer hybrids

Adaptive neuron models

Recurrent SNN architectures

Attention mechanisms for spike sequences

⚡ Hardware Deployment

🔬 Intel Loihi

🧠 SpiNNaker

⚙️ BrainScaleS

⚡ FPGA-based neuromorphic acceleration

📡 Edge Healthcare

⌚ Wearable biosignal monitoring

⚡ Low-power edge inference

📡 Continuous ECG / EEG / EMG streaming

🚨 On-device anomaly detection

🔬 Advanced Explainability

🕒 Temporal spike attribution

🧠 Neuron-level explanations

🔍 Counterfactual explanations

🧬 Modality-level explanation fusion

🌐 Multi-Modal Fusion

Future versions can combine ECG, EEG, and EMG representations into a unified multi-modal neuromorphic model.

📚 Research Contribution

NeuroPulse AI is designed around the following research contributions:

Unified multi-modal SNN pipeline for ECG, EEG, and EMG biomedical signals.

Multiple spike encoding strategies for converting continuous biosignals into event-based representations.

Comparative benchmarking between SNNs and conventional CNN / LSTM / Transformer architectures.

Explainable SNN analysis using spike attribution and signal heatmaps.

Energy and computational analysis focused on neuromorphic efficiency.

Interactive observability dashboard connecting model inference, spike behavior, explainability, and research metrics.

Software-only neuromorphic simulation that enables experimentation without dedicated neuromorphic hardware.

📌 Project Status

Status: 🚧 Research Prototype / Active Development

The core project structure includes:

🧬 Multi-modal preprocessing

⚡ Spike encoding

🧠 ECG / EEG / EMG SNN models

🧩 CNN / LSTM / Transformer baselines

🏋️ Training pipelines

📊 Benchmarking

🧮 Complexity analysis

🔍 XAI

⚙️ FastAPI backend

📡 WebSocket infrastructure

🖥️ Next.js research dashboard

📈 Visualization outputs

👩‍💻 Author

Sanz Wadibhasme

Artificial Intelligence & Data Science

GitHub: @sanz2005

⭐ Acknowledgement

This project is developed as a research-oriented exploration of:

Neuromorphic computing

Spiking Neural Networks

Biomedical signal processing

Explainable AI

Energy-aware machine learning

The project builds on publicly available biomedical datasets and open-source machine-learning technologies.

<p align="center">
  <strong>NeuroPulse AI — From Biosignals to Spikes to Explainable Healthcare Intelligence.</strong>
</p>
