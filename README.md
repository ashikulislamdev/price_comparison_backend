# 🖥️ Price Comparison Backend (FastAPI + TFLite)

This repository contains the backend implementation for the **Image-Based Price Comparison System**.

## 🚀 Features

- 🔍 Predicts product category from uploaded image using MobileNetV2 TFLite model
- 📦 Queries Amazon using Rainforest API
- 🏪 Supports Walmart, eBay, JD using mock APIs (JSON)
- ⚡ Powered by FastAPI (REST API)
- 📄 Includes both model-serving and mock data-serving services

## 📸 Demo Video

🎥 Watch the full walkthrough: [https://youtu.be/vfC0H4d8Rfc](https://youtu.be/vfC0H4d8Rfc)

## 🧠 ML Model

- Model: MobileNetV2
- Dataset: Fashion MNIST
- Converted to TFLite for fast inference
- Input size: 96x96 RGB
- Accuracy: ~93% training, ~89% validation

- python run_all.py
