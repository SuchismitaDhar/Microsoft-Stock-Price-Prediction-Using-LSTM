# 📈 Microsoft Stock Price Prediction Using LSTM
A deep learning project that predicts Microsoft (MSFT) stock prices using Long Short-Term Memory (LSTM) neural networks. The model achieves **R² score of 0.8873** on test data and generates 30-day future price forecasts.

## 🎯 Project Overview

This project implements an end-to-end time-series forecasting system for stock price prediction. It demonstrates practical application of:
- LSTM neural networks for sequential data
- Time-series data preprocessing and feature engineering
- Model optimization and performance evaluation
- Real-world problem-solving in financial data science

## 🛠️ Technology Stack

- **Programming Language**: Python 3.10.8
- **Deep Learning**: TensorFlow 2.15.0, Keras
- **Data Processing**: NumPy 1.26.4, Pandas 2.1.4
- **Visualization**: Matplotlib 3.8.2, Seaborn 0.13.0
- **Machine Learning**: Scikit-learn 1.3.2
- **Environment**: Jupyter Notebook, Anaconda

## 📂 Project Structure

```
microsoft-stock-lstm-prediction/
│
├── notebooks/
│   └── LSTM_Stock_Prediction.ipynb    # Main implementation notebook
│
├── data/
│   ├── microsoft_stock_data.csv       # Historical stock data
│   └── realtime_data.csv              # Real-time validation data
│
├── models/
│   ├── lstm_model.h5                  # Trained LSTM model
│   └── scaler.pkl                     # Data scaler
│
├── visualizations/                    # Generated plots and charts
│
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SuchismitaDhar/Microsoft-Stock-Price-Prediction-Using-LSTM.git
cd Microsoft-Stock-Price-Prediction-Using-LSTM
```

2. **Create virtual environment**
```bash
# Using conda (recommended)
conda create -n stock_pred python=3.10
conda activate stock_pred

# Or using venv
python -m venv stock_pred
source stock_pred/bin/activate  # On Windows: stock_pred\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the notebook**
```bash
jupyter notebook notebooks/Microsoft_Stock_Price_LSTM.ipynb
```

---

## 📊 Methodology

### 1. Data Preprocessing
- Loaded historical Microsoft stock data (Open, High, Low, Close, Volume)
- Handled missing values using linear interpolation
- Created datetime index for time-series analysis

### 2. Feature Engineering
Developed technical indicators:
- **Moving Averages**: SMA (20, 50, 200-day), EMA (20-day)
- **Volatility Indicators**: Bollinger Bands, rolling standard deviation
- **Momentum Indicators**: RSI, MACD, daily returns
- **Volume Metrics**: Trading volume changes

### 3. Data Preparation
```python
# Key implementation: Proper time-series handling
1. Chronological train-test split (80-20)
2. Scale training data → fit scaler
3. Transform test data using trained scaler
4. Create sequences: 60-day lookback window
```
**Critical**: Split data **before** scaling to prevent data leakage.

### 4. Model Architecture

**Univariate LSTM Model** (Best Performance):
```
Input: (60 timesteps, 1 feature)

LSTM(50) + Dropout(0.2) + return_sequences=True
LSTM(50) + Dropout(0.2) + return_sequences=False
Dense(1)

Optimizer: Adam (lr=0.001)
Loss: MSE

### 5. Training
- Epochs: 100 with Early Stopping (patience=20)
- Batch Size: 32
- Validation Split: 20%
- Callbacks: EarlyStopping, ModelCheckpoint

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | **0.8873** |
| **MAE** | $3.21 |
| **RMSE** | $4.15 |

### Key Insights
✅ Univariate model outperformed multi-feature approach  
✅ 60-day lookback window optimal for capturing patterns  
✅ Model successfully captures trend and seasonality  
✅ Predictions align closely with actual price movements  

## 🔍 Challenges & Solutions

### Challenge 1: Multi-Feature Model Underperformance
- **Problem**: Initial 14-feature model achieved R² = -0.41
- **Solution**: Switched to univariate model (Close price only)
- **Result**: R² improved to 0.8873

### Challenge 2: Data Leakage Issues
- **Problem**: Test data scaled outside [0,1] range
- **Solution**: Split data before scaling, fit scaler on train only
- **Result**: Proper scaling, no information leakage

### Challenge 3: Insufficient Real-Time Data
- **Problem**: 85 data points too small for LSTM
- **Solution**: Downloaded extended historical data (500+ days)
- **Result**: Model trained effectively with adequate samples

## 📊 Visualizations

The project includes:
- Historical price trends with technical indicators
- Training/validation loss curves
- Actual vs. predicted price comparison
- 30-day future price forecast
- Prediction error distribution analysis

## 🎓 Key Learnings

1. **Time-Series Handling**: Never shuffle time-series data; use chronological splits
2. **Feature Selection**: Simpler models often outperform complex ones
3. **Data Scaling**: Always fit scaler on training data only
4. **Sample Size**: LSTM requires 200+ data points for effective learning
5. **Hyperparameters**: Lookback window significantly impacts performance

## 🔮 Future Enhancements

- [ ] Implement ensemble methods (LSTM + GRU)
- [ ] Add sentiment analysis from financial news
- [ ] Multi-stock portfolio prediction
- [ ] Real-time API deployment
- [ ] Attention mechanisms for interpretability

## 📝 Requirements

```
numpy==1.26.4
pandas==2.1.4
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
tensorflow==2.15.0
jupyter
```

## 👤 Author

**Suchismita Dhar**  
- 📧 Email: suchisd99@gmail.com  
- 💼 LinkedIn: [Suchismita Dhar] (https://www.linkedin.com/in/suchismita-dhar99/)  
- 🐱 GitHub: [SuchismitaDhar](https://github.com/SuchismitaDhar)

## 🙏 Acknowledgments

- Microsoft Corporation for publicly available stock data
- TensorFlow team for excellent deep learning framework
- Open-source community for resources and support

## ⚠️ Disclaimer

**This project is for educational and research purposes only. This is NOT financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 📚 References
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Stock Price Prediction Research Papers](https://scholar.google.com/)

**⭐ If you found this project helpful, please consider giving it a star!**

