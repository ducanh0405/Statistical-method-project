# Statistical Methods Project - Experience & Salary Analysis

## 📊 Project Overview

This project performs comprehensive statistical analysis on the **Experience-Salary Dataset** from Kaggle. The analysis includes descriptive statistics, confidence intervals, hypothesis testing, and regression analysis using Python.

**Dataset Source:** [Experience-Salary Dataset on Kaggle](https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset/data)

---

## 🎯 Project Objectives

The project addresses the following statistical analysis tasks:

1. **Descriptive Statistics** for Experience (X) and Salary (Y)
2. **Confidence Intervals** for mean Experience and mean Salary
3. **Hypothesis Tests** for mean Experience and mean Salary
4. **Regression Analysis** to model the relationship between Experience and Salary

---

## 🔧 Requirements

### Python Libraries

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Required Libraries:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scipy` - Scientific computing and statistical tests
- `scikit-learn` - Machine learning and regression analysis

---

## 📁 Project Structure

```
Statistical-method-project/
│
├── statistical_analysis.py      # Main analysis script
├── README.md                     # Project documentation
├── Salary_Data.csv              # Dataset (download from Kaggle)
│
├── output/
│   ├── descriptive_statistics.png
│   ├── confidence_intervals.png
│   └── regression_analysis.png
```

---

## 🚀 Usage

### Step 1: Download the Dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset/data)
2. Download the dataset
3. Save it as `Salary_Data.csv` in the project directory

### Step 1.1: Data Preprocessing (tự động trong code)
- Tự động chọn 2 cột numeric đầu tiên (ưu tiên `Experience/YearsExperience` và `Salary`)
- Chuyển đổi giá trị `exp(in months)` sang **years** nếu phát hiện cột chứa chữ “month”
- Chuẩn hóa numeric, loại bỏ:
  - NaN / Inf
  - Dòng trùng lặp
  - Outlier theo quy tắc IQR (1.5 × IQR) cho cả X và Y
- In ra tóm tắt số dòng bị loại và số dòng cuối dùng cho phân tích
- Tự động tạo folder `output/` nếu chưa tồn tại

### Step 2: Run the Analysis

```bash
python statistical_analysis.py
```

The script will:
- Load and prepare the data
- Perform all statistical analyses
- Generate visualization files
- Display comprehensive results in the console

---

## 📈 Analysis Components

### 1. Descriptive Statistics

**For Experience (X) and Salary (Y):**
- Count, Mean, Median, Mode
- Standard Deviation, Variance
- Min, Max, Range
- Quartiles (Q1, Q2, Q3) and IQR
- Skewness and Kurtosis
- Coefficient of Variation
- Pearson Correlation Coefficient

**Visualizations:**
- Histograms with mean/median lines
- Box plots
- Scatter plot (Experience vs Salary)
- Q-Q plot for normality assessment

### 2. Confidence Intervals

**95% Confidence Intervals for:**
- Mean Experience (years)
- Mean Salary ($)

**Includes:**
- Sample statistics (mean, std dev, standard error)
- t-critical values
- Margin of error
- Interpretation of results

**Visualizations:**
- Error bar plots with confidence intervals

### 3. Hypothesis Tests

**One-Sample t-Tests:**
- Test for mean Experience (H₀: μ = μ₀)
- Test for mean Salary (H₀: μ = μ₀)

**Includes:**
- t-statistics and p-values
- Critical values
- Decision rules (reject/fail to reject H₀)
- Normality tests (Shapiro-Wilk)

### 4. Regression Analysis

**Simple Linear Regression:**
- Model: `Salary = β₀ + β₁ × Experience`

**Model Evaluation:**
- R² (Coefficient of Determination)
- Adjusted R²
- RMSE, MSE, MAE
- Statistical significance tests
- ANOVA table

**Residual Diagnostics:**
- Residual plots
- Normal Q-Q plot
- Scale-Location plot
- Durbin-Watson test for autocorrelation

**Visualizations:**
- Regression line with data points
- Residuals vs Fitted values
- Residual distribution histogram
- Multiple diagnostic plots

---

## 📊 Sample Output

```
================================================================================
                    EXPERIENCE-SALARY STATISTICAL ANALYSIS
                         Data Science Project - HCMIU
================================================================================

1. DESCRIPTIVE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Statistics for X (Experience):
  Count......................... 30.0000
  Mean.......................... 5.3133
  Median........................ 4.7000
  ...

2. CONFIDENCE INTERVALS (95%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ 95% CI for Mean Experience: [4.5521, 6.0746]
✨ 95% CI for Mean Salary: [54199.45, 68894.22]

3. HYPOTHESIS TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  t-statistic: 2.1543
  p-value: 0.0397
  ✅ Reject H₀ (significant at α=0.05)

4. REGRESSION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Regression Equation:
  Salary = 25792.20 + 9449.96 × Experience
  
  R² = 0.9569
  p-value < 0.0001
  ✅ Model is statistically significant
```

---

## 🎓 Statistical Interpretations

### Key Findings (Example):

1. **Strong Positive Correlation**: Experience and Salary show high positive correlation
2. **Linear Relationship**: The regression model explains ~95% of salary variation
3. **Significant Predictors**: Experience significantly predicts salary (p < 0.001)
4. **Salary Increase**: Each additional year of experience increases salary by ~$9,450

---

## 🔬 Statistical Methods Used

- **Descriptive Statistics**: Central tendency, dispersion, shape measures
- **Inferential Statistics**: t-tests, confidence intervals
- **Regression Analysis**: OLS (Ordinary Least Squares)
- **Diagnostic Tests**: Normality tests, residual analysis
- **ANOVA**: Analysis of variance for regression

---

## 📝 Notes

- All statistical tests use α = 0.05 significance level (customizable)
- Confidence intervals calculated at 95% confidence level (customizable)
- The script includes comprehensive error handling and data validation
- Visualizations are saved as high-resolution PNG files (300 DPI)

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

---

## 📄 License

This project is for educational purposes as part of the Data Science curriculum at HCMIU.

---

## 🔗 References

- Dataset: [Kaggle - Experience Salary Dataset](https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset/data)
- Statistical Methods: Standard inferential statistics and regression analysis
- Python Documentation: [SciPy](https://docs.scipy.org/), [Scikit-learn](https://scikit-learn.org/)

---

**Happy Analyzing! 📊✨**
