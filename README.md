# Tesla Stock Price Prediction Using LSTM Neural Networks 📈🤖

**Machine Learning Internship - Minor Project**  
**October-December 2024 Batch**  
**Project Date: 16.11.2024**

A comprehensive deep learning project that predicts Tesla (TSLA) stock prices using Long Short-Term Memory (LSTM) neural networks with advanced time series analysis and technical indicators.

## 🎯 Project Overview

This project implements a sophisticated stock price prediction system using deep learning techniques to forecast Tesla's stock prices. The model leverages historical stock data spanning from 2010 to 2024, incorporating moving averages and LSTM neural networks to capture complex temporal patterns in financial markets.

## 📊 Dataset Information

- **Stock Symbol**: TSLA (Tesla Inc.)
- **Time Period**: June 29, 2010 - January 19, 2024
- **Total Data Points**: 3,413 trading days
- **Data Source**: Yahoo Finance (yfinance API)
- **Training Data**: 84% (2,866 days)
- **Testing Data**: 16% (547 days)

### Key Features
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price (target variable)
- **Adj Close**: Adjusted closing price
- **Volume**: Number of shares traded

## 🛠️ Technologies & Libraries Used

### Core Libraries
```python
import numpy as np              # Numerical computing
import pandas as pd             # Data manipulation
import matplotlib.pyplot as plt # Data visualization
import yfinance as yf          # Financial data API
```

### Machine Learning & Deep Learning
```python
from sklearn.preprocessing import MinMaxScaler  # Data normalization
from keras.layers import Dense, Dropout, LSTM  # Neural network layers
from keras.models import Sequential             # Model architecture
```

## 🧠 Model Architecture

### LSTM Neural Network Structure
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ LSTM Layer 1 (50 units)             │ (None, 100, 50)             │          10,400 │
│ Dropout (0.2)                       │ (None, 100, 50)             │               0 │
│ LSTM Layer 2 (60 units)             │ (None, 100, 60)             │          26,640 │
│ Dropout (0.3)                       │ (None, 100, 60)             │               0 │
│ LSTM Layer 3 (80 units)             │ (None, 100, 80)             │          45,120 │
│ Dropout (0.4)                       │ (None, 100, 80)             │               0 │
│ LSTM Layer 4 (120 units)            │ (None, 120)                 │          96,480 │
│ Dropout (0.5)                       │ (None, 120)                 │               0 │
│ Dense Output Layer                   │ (None, 1)                   │             121 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```

### Model Specifications
- **Total Parameters**: 536,285 (2.05 MB)
- **Trainable Parameters**: 178,761 (698.29 KB)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Sequence Length**: 100 days lookback
- **Epochs**: 50
- **Batch Size**: 32

## 📈 Technical Analysis Features

### Moving Averages Implementation
- **100-Day Moving Average**: Short-term trend indicator
- **200-Day Moving Average**: Long-term trend indicator
- **Price Visualization**: Comparative analysis with actual closing prices

### Data Preprocessing Pipeline
1. **Data Collection**: Yahoo Finance API integration
2. **Feature Engineering**: Moving averages calculation
3. **Data Cleaning**: Missing value handling
4. **Normalization**: MinMaxScaler (0-1 range)
5. **Sequence Creation**: 100-day sliding windows
6. **Train-Test Split**: 84%-16% ratio

## 🔍 Model Training Results

### Training Performance
- **Training Duration**: 50 epochs
- **Final Training Loss**: ~0.001 (very low MSE)
- **Training Progress**: Consistent loss reduction from 0.0139 to 0.001
- **Convergence**: Achieved around epoch 25-30

### Key Training Insights
- **Epoch 1**: Loss = 0.0139 (initial high loss)
- **Epoch 25**: Loss = 0.0011 (significant improvement)
- **Epoch 50**: Loss = 0.0010 (final convergence)
- **Training Stability**: No overfitting observed

## 📊 Prediction Results & Visualization

### Model Performance Analysis
The prediction visualization shows:
- **Red Line**: LSTM Predicted Prices
- **Green Line**: Actual Tesla Stock Prices
- **Correlation**: Strong alignment between predicted and actual prices
- **Trend Capture**: Successfully captures major price movements and volatility

### Key Observations
✅ **Strengths**:
- Accurately captures overall price trends
- Follows major price movements and reversals
- Handles high volatility periods well
- Maintains directional accuracy

⚠️ **Areas for Improvement**:
- Some lag in capturing sudden price spikes
- Minor deviations during extreme volatility
- Could benefit from additional technical indicators

## 🗂️ Project Structure

```
STOCK PREDICTION FOR MINOR PROJECT.ipynb
├── 📥 Data Collection (yfinance)
├── 📊 Exploratory Data Analysis
│   ├── Price visualization
│   ├── Moving averages analysis
│   └── Trend identification
├── 🔧 Data Preprocessing
│   ├── Data cleaning
│   ├── MinMax scaling
│   └── Sequence preparation
├── 🧠 LSTM Model Development
│   ├── Architecture design
│   ├── Layer configuration
│   └── Hyperparameter tuning
├── 🎯 Model Training
│   ├── 50 epochs training
│   ├── Loss monitoring
│   └── Performance tracking
├── 📈 Prediction & Evaluation
│   ├── Test data prediction
│   ├── Inverse scaling
│   └── Results visualization
└── 📋 Analysis & Insights
```

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install numpy pandas matplotlib yfinance
pip install tensorflow keras scikit-learn
pip install jupyter notebook
```

### Running the Project

1. **Clone the repository:**
```bash
git clone [your-repository-url]
cd tesla-stock-prediction
```

2. **Launch Jupyter Notebook:**
```bash
jupyter notebook "STOCK PREDICTION FOR MINOR PROJECT.ipynb"
```

3. **Execute cells sequentially** to:
   - Download Tesla stock data
   - Perform technical analysis
   - Train the LSTM model
   - Generate predictions

## 📊 Technical Indicators Analysis

### Moving Averages Insights
- **100-Day MA**: Captures short-term momentum
- **200-Day MA**: Identifies long-term trends
- **Golden Cross**: When 100-day MA crosses above 200-day MA (bullish signal)
- **Death Cross**: When 100-day MA crosses below 200-day MA (bearish signal)

### Price Action Analysis
- **Volatility Periods**: 2020-2021 showed extreme volatility
- **Trend Reversals**: Successfully identified major turning points
- **Support/Resistance**: Moving averages act as dynamic support/resistance levels

## 🎯 Model Applications

### 📈 **Investment Decision Support**
- Short-term price direction prediction
- Risk assessment for portfolio management
- Entry/exit point identification

### 📊 **Financial Analysis**
- Market trend analysis
- Volatility forecasting
- Technical indicator validation

### 🤖 **Algorithmic Trading**
- Automated trading signal generation
- Risk management integration
- Real-time prediction capabilities

## 🔮 Future Enhancements

### Technical Improvements
- [ ] **Additional Features**: Volume, RSI, MACD indicators
- [ ] **Ensemble Methods**: Combine multiple LSTM models
- [ ] **Attention Mechanisms**: Transformer-based architectures
- [ ] **Hyperparameter Optimization**: Grid search for optimal parameters

### Advanced Features
- [ ] **Multi-Stock Prediction**: Portfolio-wide forecasting
- [ ] **Sentiment Analysis**: News and social media integration
- [ ] **Real-time Streaming**: Live market data integration
- [ ] **Risk Metrics**: VaR, Sharpe ratio calculations

### Deployment Options
- [ ] **Web Application**: Flask/Django deployment
- [ ] **API Development**: RESTful prediction service
- [ ] **Mobile App**: Real-time stock predictions
- [ ] **Cloud Integration**: AWS/GCP deployment

## ⚠️ Important Disclaimer

**This project is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with qualified financial advisors and conduct thorough research before making investment choices.**

## 📊 Key Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Training Loss** | 0.001 | Final MSE after 50 epochs |
| **Model Size** | 2.05 MB | Total model parameters |
| **Training Time** | ~30 min | 50 epochs on standard hardware |
| **Prediction Accuracy** | High | Strong correlation with actual prices |
| **Data Coverage** | 14 years | Comprehensive historical analysis |

## 🏆 Project Achievements

### ✅ **Successfully Implemented**
- **LSTM Architecture**: 4-layer deep neural network
- **Technical Analysis**: Moving averages integration
- **Data Pipeline**: Automated data collection and preprocessing
- **Visualization**: Professional prediction charts
- **Model Training**: Convergent training with low loss

### 📈 **Learning Outcomes**
- **Deep Learning**: LSTM neural network implementation
- **Financial Markets**: Technical analysis understanding
- **Time Series**: Sequential data modeling
- **Python Programming**: Advanced library usage
- **Data Science**: End-to-end ML project lifecycle

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

### Code Improvements
1. **Model Enhancement** - New architectures or optimizations
2. **Feature Engineering** - Additional technical indicators
3. **Visualization** - Enhanced charting and analysis
4. **Documentation** - Code comments and explanations

### Research Extensions
1. **Comparative Studies** - Different stock symbols
2. **Model Comparison** - LSTM vs GRU vs Transformer
3. **Performance Analysis** - Detailed accuracy metrics
4. **Market Analysis** - Sector-wise predictions

## 📧 Contact Information

**Developer**: Arun Geethan B K  
**Email**: arungeethan3474@gmail.com 
**Project**: Machine Learning Internship Minor Project  
**Batch**: October-December 2024  
**Date**: November 16, 2024  

**Project Repository**: (https://github.com/arungeethanbk/JupyterNotebook/blob/main/STOCK%20PREDICTION%20FOR%20MINOR%20PROJECT.ipynb)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing comprehensive stock market data
- **TensorFlow/Keras** community for excellent deep learning frameworks
- **Machine Learning Internship Program** for project guidance
- **Tesla Inc.** for being an interesting and volatile stock to analyze
- **Open Source Community** for invaluable libraries and tools

---

⭐ **If this project helped you understand LSTM stock prediction, please give it a star!** ⭐

**📈 Democratizing AI for Financial Analysis 💰**



# Employee Salary Prediction

A machine learning project that predicts employee salaries based on various factors including age, gender, education level, job title, and years of experience using Linear Regression.

## 📊 Dataset Overview

The dataset contains employee information with the following features:
- **Age**: Employee's age
- **Gender**: Male/Female
- **Education Level**: Bachelor's, Master's, PhD
- **Job Title**: Various positions (174 unique job titles)
- **Years of Experience**: Professional experience in years
- **Salary**: Target variable (annual salary in USD)

**Dataset Statistics:**
- Original size: 375 records
- After cleaning: 324 records
- Features: 6 columns

## 🔍 Data Analysis & Preprocessing

### Data Cleaning
- Removed 50 duplicate records
- Handled missing values (2 records with null values)
- Final dataset: 324 clean records

### Key Insights
- **Age Distribution**: Mean age of 37.4 years (range: 23-53)
- **Experience**: Average 10.1 years of experience (range: 0-25)
- **Salary Range**: $350 - $250,000 (mean: $99,986)
- **Education Split**: 
  - Bachelor's: 191 employees (59%)
  - Master's: 91 employees (28%)
  - PhD: 42 employees (13%)

### Correlation Analysis
Strong correlations found between:
- Age ↔ Experience: 0.979
- Experience ↔ Salary: 0.924
- Age ↔ Salary: 0.917

## 🛠️ Feature Engineering

### Encoding Categorical Variables
- **Gender**: Label encoded (Female: 0, Male: 1)
- **Education Level**: Label encoded (Bachelor's: 0, Master's: 1, PhD: 2)
- **Job Title**: Label encoded (174 unique titles)

### Feature Scaling
- Applied StandardScaler to Age and Years of Experience
- Normalized features for better model performance

## 🤖 Machine Learning Model

### Algorithm Used
**Linear Regression** - Chosen for its interpretability and effectiveness with continuous target variables.

### Model Performance
- **Accuracy (R² Score)**: 89.11%
- **Mean Absolute Error**: $10,570.79
- **Root Mean Squared Error**: $14,344.13

### Model Coefficients
The model learned the following feature importance:
- Age (scaled): $20,182 impact per unit
- Experience (scaled): $19,204 impact per unit
- Education Level: $15,423 impact per unit
- Gender: $7,389 impact per unit
- Job Title: $19.58 impact per unit

## 📈 Model Evaluation

### Train-Test Split
- Training Set: 259 samples (80%)
- Testing Set: 65 samples (20%)
- Random State: 42 (for reproducibility)

### Performance Metrics
The model demonstrates strong predictive capability with nearly 89% accuracy, indicating reliable salary predictions based on the input features.

## 🚀 Usage Example

### Making Predictions
```python
# Example prediction for:
# Age: 50, Gender: Female, Education: PhD, Job: Director, Experience: 20 years
predicted_salary = model.predict([[scaled_age, 0, 2, 22, scaled_experience]])
# Result: $250,727
```

## 📋 Requirements

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 🏃‍♂️ How to Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd salary-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Prepare the dataset**
   - Ensure `Dataset09-Employee-salary-prediction.csv` is in the project directory

4. **Run the Jupyter notebook**
   ```bash
   jupyter notebook salary_prediction.ipynb
   ```

## 📁 Project Structure

```
salary-prediction/
│
├── salary_prediction.ipynb          # Main analysis notebook
├── Dataset09-Employee-salary-prediction.csv  # Dataset
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

## 🔮 Future Improvements

- Experiment with advanced algorithms (Random Forest, XGBoost)
- Feature selection techniques to identify most important predictors
- Cross-validation for more robust model evaluation
- Hyperparameter tuning for optimal performance
- Deployment as a web application

## 📊 Visualizations Included

- Correlation heatmap of numerical features
- Distribution plots for education levels and gender
- Box plots for outlier detection
- Histogram analysis of age and salary distributions

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---
Author: Arun Geethan B K
Repository: JupyterNotebook/Salary_prediction.ipynb

**Note**: This model is trained on a specific dataset and should be used for educational purposes. Real-world salary predictions may require more comprehensive data and domain expertise.


# Advanced Salary Predictions

A comprehensive machine learning project for predicting job salaries using multiple regression techniques and advanced feature engineering on a large-scale dataset of 1 million job records.

## 🌟 Project Highlights

- **Large-Scale Dataset**: 1 million job records with comprehensive feature analysis
- **Multiple ML Algorithms**: Linear Regression, Ridge Regression, Polynomial Features, Random Forest
- **Advanced Techniques**: Cross-validation, Grid Search, Pipeline optimization
- **Professional Data Pipeline**: Modular functions for data loading, cleaning, and analysis

## 📊 Dataset Overview

### Dataset Scale
- **Training Features**: 1,000,000 records × 8 features
- **Training Targets**: 1,000,000 salary records  
- **Test Features**: 1,000,000 records for predictions
- **Total Memory Usage**: ~137MB across all datasets

### Features Description

#### Categorical Variables
- **companyId**: Company identifier (63 unique companies)
- **jobType**: Position level (8 types: CEO, CFO, CTO, VP, Manager, Senior, Junior, Janitor)
- **degree**: Education level (5 levels: Doctoral, Masters, Bachelors, High School, None)
- **major**: Field of study (9 majors: Engineering, Business, Computer Science, etc.)
- **industry**: Business sector (7 industries: Web, Auto, Finance, Education, Oil, Health, Service)

#### Numerical Variables
- **yearsExperience**: Professional experience (0-24 years)
- **milesFromMetropolis**: Distance from major city (0-99 miles)

#### Target Variable
- **salary**: Annual salary in thousands (range: $17K - $301K)

## 🔍 Exploratory Data Analysis

### Data Quality Assessment
- **Missing Values**: No missing data across all features
- **Invalid Records**: 5 records with $0 salary removed
- **Final Clean Dataset**: 999,995 records
- **Duplicate Handling**: Removed based on jobId uniqueness

### Key Statistical Insights

#### Salary Distribution
- **Mean Salary**: $116,062
- **Median Salary**: $114,000
- **Standard Deviation**: $38,717
- **Distribution**: Approximately symmetric (skewness: 0.35)
- **Outlier Threshold**: Salaries > $220.5K or < $8.5K

#### Feature Distributions
- **Job Types**: Fairly balanced distribution across all levels
- **Education**: High School/None (47%), Bachelor's (18%), Advanced degrees (35%)
- **Experience**: Normal distribution with mean 12 years
- **Location**: Uniform distribution from metropolitan areas

### Outlier Analysis
- **High Salary Outliers**: 7,117 records above $220.5K
- **Interesting Finding**: 20 Junior positions with salaries > $220K (likely high-skill tech roles)
- **No Low Salary Outliers**: All salaries above $8.5K threshold

## 🛠️ Advanced Feature Engineering

### Data Preprocessing Pipeline
```python
# Modular approach with custom functions
- load_f(): Efficient CSV loading
- clean_d(): Remove duplicates and invalid salaries  
- Feature encoding for categorical variables
- Standardization for numerical features
```

### Visualization Functions
- **Scatter Analysis**: `scatter_data()` for relationship exploration
- **Regression Plots**: `reg_data()` with trend lines
- **Residual Analysis**: `res_data()` for model validation
- **Distribution Comparison**: `dis_data()` for feature comparison
- **Categorical Analysis**: `rel_cat()` with violin plots and box plots

## 🤖 Machine Learning Pipeline

### Algorithms Implemented
1. **Linear Regression**: Baseline model for interpretability
2. **Ridge Regression**: L2 regularization for overfitting prevention
3. **Polynomial Features**: Capturing non-linear relationships
4. **Random Forest Regressor**: Ensemble method for robust predictions

### Model Validation Strategy
- **Cross-Validation**: K-fold validation for robust performance metrics
- **Grid Search**: Hyperparameter optimization
- **Stratified Sampling**: Ensuring representative train/test splits
- **Pipeline Integration**: Seamless preprocessing and modeling

### Advanced Techniques
- **Feature Scaling**: StandardScaler for numerical features
- **Polynomial Features**: Higher-order feature interactions
- **Model Serialization**: joblib for model persistence
- **Residual Analysis**: Comprehensive error pattern analysis

## 📈 Model Performance Metrics

### Evaluation Framework
- **Mean Squared Error (MSE)**: Primary regression metric
- **Cross-Validation Scores**: Robust performance estimation
- **Residual Analysis**: Error pattern identification
- **Feature Importance**: Random Forest feature ranking

### Performance Comparison
Multiple models evaluated with cross-validation to ensure generalizability and prevent overfitting.

## 🔧 Technical Implementation

### Required Libraries
```python
pandas              # Data manipulation
scikit-learn        # Machine learning algorithms
numpy              # Numerical computations
seaborn            # Statistical visualizations
matplotlib         # Plotting and charts
scipy              # Statistical functions
```

### Key Technical Features
- **Memory Efficient**: Optimized for large dataset processing
- **Modular Design**: Reusable functions for different datasets
- **Error Handling**: Robust data validation and cleaning
- **Scalable Architecture**: Can handle datasets of varying sizes

## 🚀 Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/arungeethanbk/JupyterNotebook.git
cd JupyterNotebook
pip install pandas scikit-learn numpy seaborn matplotlib scipy
```

### 2. Data Preparation
```bash
# Ensure these files are in the data/ directory:
# - train_features.csv
# - train_salaries.csv  
# - test_features.csv
```

### 3. Run the Analysis
```bash
jupyter notebook Salary_Predictions_2.ipynb
```

## 📁 Project Structure

```
JupyterNotebook/
│
├── Salary_Predictions_2.ipynb    # Main analysis notebook
├── data/
│   ├── train_features.csv        # Training features (1M records)
│   ├── train_salaries.csv        # Training targets
│   └── test_features.csv         # Test features for predictions
├── README.md                     # This documentation
└── requirements.txt              # Python dependencies
```

## 🎯 Key Insights & Findings

### Business Intelligence
- **Experience Premium**: Strong positive correlation between experience and salary
- **Education Impact**: Advanced degrees show significant salary premiums
- **Geographic Factor**: Distance from metropolis negatively affects salary
- **Industry Variations**: Finance and Oil sectors show higher average salaries
- **Position Hierarchy**: Clear salary progression from Junior to Executive roles

### Data Science Insights
- **Linear Relationships**: Strong linear correlation between key numerical features
- **Feature Interactions**: Polynomial features may capture experience-education interactions
- **Outlier Patterns**: High-salary junior positions indicate specialized skill premiums
- **Model Complexity**: Ensemble methods likely to outperform linear models

## 🔮 Advanced Features & Future Work

### Current Advanced Implementation
- **Multi-Algorithm Comparison**: Systematic evaluation of different regression techniques
- **Feature Engineering Pipeline**: Automated preprocessing with sklearn Pipeline
- **Cross-Validation Framework**: Robust model validation methodology
- **Residual Analysis**: Comprehensive error pattern analysis

### Potential Enhancements
- **Deep Learning**: Neural networks for complex pattern recognition
- **Feature Selection**: Automated feature importance ranking
- **Hyperparameter Tuning**: Bayesian optimization for better parameters
- **Model Ensemble**: Combining multiple algorithms for improved accuracy
- **Real-time Prediction API**: Flask/FastAPI deployment

## 📊 Visualizations & Analysis

### Comprehensive EDA
- **Correlation Heatmaps**: Feature relationship analysis
- **Distribution Plots**: Salary and feature distributions
- **Regression Plots**: Linear relationship visualization
- **Residual Plots**: Model assumption validation
- **Box Plots**: Categorical feature impact analysis
- **Violin Plots**: Distribution shape comparison

## 🤝 Contributing

This project welcomes contributions! Areas for enhancement:
- Additional feature engineering techniques
- New regression algorithms implementation
- Advanced visualization methods
- Performance optimization for larger datasets
- Model interpretability improvements

## 📄 License

Open source project available under the MIT License. Feel free to use, modify, and distribute.

## 🏆 Project Impact

This project demonstrates enterprise-level data science capabilities:
- **Scalability**: Handles million-record datasets efficiently
- **Methodology**: Professional ML pipeline implementation  
- **Business Value**: Actionable insights for HR and compensation decisions
- **Technical Depth**: Advanced statistical and ML techniques
- **Reproducibility**: Well-documented, modular code structure

---

**Author**: [Arun Geethan BK](https://github.com/arungeethanbk)  
**Repository**: [JupyterNotebook/Salary_Predictions_2.ipynb](https://github.com/arungeethanbk/JupyterNotebook/blob/main/Salary_Predictions_2.ipynb)

*This project showcases advanced machine learning techniques applied to real-world salary prediction challenges using large-scale data.*
