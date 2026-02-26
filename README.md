# 🛠️ Technology Stack used for

- **Programming Language**: Python 3.10.8
- **Deep Learning**: TensorFlow 2.15.0, Keras
- **Data Processing**: NumPy 1.26.4, Pandas 2.1.4
- **Visualization**: Matplotlib 3.8.2, Seaborn 0.13.0
- **Machine Learning**: Scikit-learn 1.3.2
- **Environment**: Jupyter Notebook, Anaconda
# 1. Heart Disease Prediction using Logistic Regression
The model serves as a foundation for clinical decision support systems and showcases the potential of predictive analytics in improving patient outcomes. Future enhancements can further improve accuracy and make this tool production-ready for real-world healthcare applications.

## 🎯 Project Overview
-**Machine Learning**: Algorithm selection, training, and evaluation of obtained medical tests
-**Data Science**: End-to-end pipeline from raw data to inference
-**Healthcare Analytics**: Domain-specific problem-solving and rendering reusable results
-**Communication**: Translating technical results for stakeholders 
-**Python Programming**:  Coding optimization and library utilization 
-**Key Libraries used**: Scikit-learn, Pandas, Matplotlib, Seaborn Academic 
-**References**: Logistic Regression for Binary Classification, Binary Cross-Entropy, Loss Function, ROC Curve Analysis and AUC Interpretation, Model Evaluation Metrics in Healthcare

# 2.Microsoft Stock Price Prediction
This project successfully demonstrated end-to-end implementation of a machine learning solution for financial time series forecasting. The Linear Regression model achieved exceptional performance with 99.86% accuracy, proving that proper feature engineering and data preparation can yield superior results even with simpler algorithms.

## 📊 Methodology: 
Rigorous time series handling preventing data leakage
Comprehensive technical indicator engineering
Thorough exploratory data analysis
Proper train-test validation methodology
Recognition of model limitations (Random Forest rejection)
Real-world applicability with 30-day forecasting capability

**The project developed foundations in**:
Machine learning model development and evaluation
Financial data analysis and technical indicators
Time series forecasting methodologies
Python programming and data science libraries
Problem-solving and troubleshooting complex issues

**This project is for educational and research purposes only. This is NOT financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

# 3. Air Quality Prediction using ML
This Air Quality Prediction project successfully demonstrated the application of machine learning to environmental forecasting. 

## Work Flow :
-**Data Processing**: Successfully cleaned and preprocessed complex environmental data with missing values and temporal structure
-**Model Development**: Implemented FbProphet forecasting approaches, demonstrating thorough understanding of ML algorithms
-**Practical Forecasting**: Developed 30-day forward prediction capability with confidence intervals
-**Technical Documentation**: Created comprehensive documentation suitable for stakeholders at all technical levels

### Focused on:
Problem-Solving: Identified root causes of poor performance and implemented effective solutions
Technical Proficiency: Mastered multiple ML libraries and frameworks
Analytical Thinking: Evaluated models using appropriate metrics and made data-driven decisions
Communication: Translated technical findings into actionable insights
Attention to Detail: Ensured data quality and model validity throughout the process

### Impact:
This project provides a foundation for operational air quality forecasting systems that can inform public health decisions, support environmental policy, and improve quality of life for urban populations. The methodologies developed are scalable and applicable to other environmental prediction challenges.

# 4. Air Quality Prediction using Deep Learning 
This project successfully implemented advanced deep learning techniques for environmental data analysis and forecasting. The work involved comprehensive data preprocessing, feature engineering, building multiple LSTM architectures, and generating reliable 30-day future predictions. The final univariate LSTM model achieved an exceptional R² score of 0.93447 on the test set, demonstrating high accuracy in predicting relative humidity patterns. The project showcased proficiency in handling time-series data, neural network architecture design, and solving real-world forecasting challenges.

It demonstrated advanced capabilities in deep learning and time-series forecasting. Starting from raw environmental sensor data with multiple quality issues, the project progressed through comprehensive preprocessing, sophisticated feature engineering, and culminated in building two high-performance LSTM models.

## Key Achievements:
Good Predictive Performance: Achieved R² scores of 0.852 (multivariate) and 0.934 (univariate), significantly outperforming traditional machine learning approaches and Prophet models tested earlier.
Solved Complex Technical Challenges: Successfully diagnosed and resolved prediction collapse issue in multivariate forecasting by implementing a dedicated univariate LSTM approach.Developed complete end-to-end pipeline from data ingestion to 30-day forecasting with proper error handling and validation.

Practical Forecasting Capability: Generated reliable 720-hour (30-day) hourly forecasts with realistic daily patterns and natural variation, suitable for real-world deployment.

**Skills Demonstrated**:
Advanced Python programming and data manipulation
Deep learning framework expertise (TensorFlow/Keras)
Time-series analysis and forecasting methodologies
Problem-solving and debugging complex ML systems
Production-quality code development and documentation
Data visualization and results communication

**Impact**
This work provides a foundation for operational air quality forecasting systems that can inform public health decisions, support environmental policy development, and improve quality of life for urban populations. The methodologies developed are transferable to other environmental prediction challenges including temperature, pollutant concentrations, and weather forecasting.

# 5.Microsoft Stock Price Prediction using Deep Learning-
This project involved developing a Long Short-Term Memory (LSTM) deep learning model for predicting Microsoft Corporation (MSFT) stock prices. The work encompassed complete data science pipeline implementation—from data acquisition and preprocessing through model development, evaluation, and deployment considerations. The project demonstrated practical application of time series forecasting, deep learning architecture design, and systematic troubleshooting of machine learning challenges.
Key Achievement: Successfully developed a univariate LSTM model achieving R² score of 0.8873 on historical data, demonstrating strong predictive capability for stock price forecasting.

## Libraries/Packages used
TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
**LSTM Deep Learning Model**-Funnel architecture progressively compresses temporal information from 128→64→32→16 units, with dropout for regularization.

### CRITICAL FAILURES ENCOUNTERED & RESOLUTION STRATEGIES:
Catastrophic Model Degradation Through Repeated Training Problem Identification:

After running training cell three times consecutively, R² degraded to negative values
Model performance deteriorated with each training iteration
**Root cause**: Accumulative overfitting through sequential training sessions without model reset
**Technical Analysis**:
Demonstrated deep understanding that calling fitting the model multiple times continues training from the previous state rather than starting fresh. 
**Each iteration caused the model to**:
Memorize training data patterns increasingly
Lose generalization capability
Optimize for training set at the expense of validation performance
Eventually perform worse than baseline prediction (mean)

### Skills Demonstrated:
Deep learning lifecycle management
Understanding of model training mechanics
Debugging complex ML behavior
Implementing software engineering best practices (factory pattern for model creation)

Successfully designed, trained, and evaluated LSTM neural networks for time series forecasting, achieving 88.7% variance explanation (R² = 0.8873) on stock price prediction.
**Advanced Problem-Solving**: Systematically diagnosed and resolved 6 critical technical challenges including data scaling issues, model architecture bottlenecks, overfitting through repeated training, and insufficient data scenarios.

Most importantly, this project demonstrated the critical requirement of honest assessment: recognizing when insufficient data (85 points for deep learning) requires either data acquisition or alternative approaches, rather than forcing suboptimal solutions.
The journey from multiple failures to a successful R² = 0.8873 model exemplifies the iterative, persistent approach required for production machine learning development.

# 6.Bitcoin Price Prediction using XGBoost, Random Forest and Deep Learning 
This project develops and evaluates multiple machine learning models for predicting Bitcoin price trends based on historical data. Through systematic experimentation with XGBoost, Random Forest, and LSTM (Long Short-Term Memory) neural networks, we identified key challenges in cryptocurrency price prediction and developed robust solutions.

**Libraries/Packages used** - TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

**Key Findings**
Univariate LSTM achieved the best performance with R² = 0.5528 on test data
Traditional ML models (XGBoost, Random Forest) suffered from severe overfitting despite extensive regularization
Feature engineering and data preprocessing strategies critically impact model performance
Cryptocurrency data presents unique challenges due to high volatility and non-stationary distributions

### Key technical indicators used in this project:
**Moving Averages (SMA, EMA)**: Trend identification
**RSI (Relative Strength Index)**: Momentum measurement
**Bollinger Bands**: Volatility assessment
**Volume Analysis**: Market activity tracking

## Machine Learning Approaches-
Traditional ML
XGBoost: Gradient boosting framework, excellent for structured data
Random Forest: Ensemble method, robust to overfitting

## Deep Learning:
LSTM: Specialized RNN architecture for sequential data, captures temporal dependencies

## Final Model: 
Univariate LSTM with 3-layer architecture, achieving reliable predictions for next-day Bitcoin closing prices.
Used different preprocessing approaches optimized for each model type: sequence-first for LSTM to maintain temporal continuity and split-first for XGBoost to prevent data leakage. This demonstrates understanding of model-specific requirements.

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

## ⚠️ Disclaimer

**This project is for educational and research purposes only. Past performance does not guarantee future results.**

## 📚 References
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Stock Price Prediction Research Papers](https://scholar.google.com/)

**⭐ If you found these projects helpful, please consider giving it a star!**

