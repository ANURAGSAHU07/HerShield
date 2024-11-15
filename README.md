# 🛡️ HERSHIELD - AI-Driven Women's Safety Analytics Software
![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)
![NVIDIA](https://img.shields.io/badge/NVIDIA-CUDA-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🔧 Environment Setup

### Using Conda (Recommended)
1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create environment from the provided YAML file:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate hershield
```

4. Verify installation:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import mediapipe; print(mediapipe.__version__)"
```

### Alternative Setup (using pip)
If you prefer not to use conda, install dependencies using pip:
```bash
python -m venv hershield-env
source hershield-env/bin/activate  # On Windows: hershield-env\Scripts\activate
pip install -r requirements.txt
```

## 💡 Overview
HERSHIELD is an innovative, AI-powered surveillance solution designed to enhance women's safety through real-time monitoring of CCTV feeds. The system detects risky situations, like being alone at night or surrounded by men, and triggers instant alerts to ensure rapid police response.

### 🔍 Problem Statement
Women's safety in public spaces is a significant issue today. HERSHIELD addresses this by combining advanced AI algorithms with continuous CCTV analysis, empowering authorities to act before harm occurs.

## 🚀 Features
- 🎥 **24/7 Video Monitoring**: Continuously monitors CCTV feeds for risk detection
- 👩‍🦰 **Gender Classification**: Utilizes RCNN to identify individuals' gender and assess risk factors
- ⚠️ **Real-Time Threat Detection**: Tracks male-to-female ratios, especially at night
- 📊 **Risk Score Calculation**: Generates risk scores based on time, location, and environmental factors
- 🔥 **Hotspot Identification**: Pinpoints high-risk zones through historical and real-time data
- ✋ **Gesture Recognition**: Detects distress signals such as frantic waving or SOS gestures
- 🚨 **Instant Alerts**: Sends immediate notifications to authorities

## 🛠️ Tech Stack
- **Python** 🐍: Primary language
- **PyTorch** 🔥: Neural network training and real-time model updates
- **Mobile VNet** 🖼️: Detailed feature maps from CCTV frames
- **Finetuned YOLO** 👤: Gender-based classification
- **Vision Transformers (ViTs)** ⚡: High-level feature extraction
- **MediaPipe** 🎥: Multimodal ML pipelines
- **CUDA-Enabled GPUs** ⚙️: Accelerated video processing

## ⚙️ How It Works
1. **Frame Preprocessing**: CCTV frames are resized and normalized (640x640x3)
2. **Feature Extraction**: CSP-Darknet 53 generates detailed feature maps
3. **Gender Detection**: RCNN classifies individuals with bounding boxes
4. **Risk Assessment**: Calculates risk scores based on multiple factors
5. **Hotspot Detection**: Tracks potential danger zones
6. **Alerting System**: Sends immediate notifications for detected threats

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- CUDA-enabled GPU
- Anaconda or Miniconda (recommended)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/hershield.git
cd hershield
```

2. Set up the environment (choose one method):
```bash
# Using conda (recommended):
conda env create -f env.yml
conda activate hershield

# OR using pip:
python -m venv hershield-env
source hershield-env/bin/activate  # On Windows: hershield-env\Scripts\activate
pip install -r requirements.txt
```

3. Run the system:
```bash
python main.py
```

## ✨ Impact and Benefits
- 🛡️ **Women's Safety First**: Reduces crime rates through rapid detection
- 👮 **Assists Law Enforcement**: Optimizes patrol resources
- 💰 **Cost-Effective**: Leverages existing CCTV infrastructure
- 📈 **Highly Scalable**: Expandable to various locations
- ⚖️ **Promotes Gender Equality**: Creates safer public spaces

## 📧 Contact
- **Team**: HERSHIELD
- **Email**: anuragsahu4328@gmail.com

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
