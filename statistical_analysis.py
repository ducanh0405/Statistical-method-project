"""Statistical Analysis for Experience-Salary Dataset

This script performs comprehensive statistical analysis on the Experience-Salary dataset:
1. Descriptive statistics for X (Experience) and Y (Salary)
2. Confidence Intervals for mean X and mean Y
3. Hypothesis tests for mean X and mean Y
4. Regression analysis

Author: Data Science Student - HCMIU
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class ExperienceSalaryAnalysis:
    """Class to perform statistical analysis on Experience-Salary dataset"""
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the data"""
        print("="*80)
        print("DATA OVERVIEW")
        print("="*80)
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumn names: {self.df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        # Assuming columns are 'Experience' and 'Salary' or similar
        # Adjust column names based on actual dataset
        if 'Experience' in self.df.columns and 'Salary' in self.df.columns:
            self.x = self.df['Experience'].values
            self.y = self.df['Salary'].values
        elif 'YearsExperience' in self.df.columns and 'Salary' in self.df.columns:
            self.x = self.df['YearsExperience'].values
            self.y = self.df['Salary'].values
        else:
            # Use first two numerical columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.x = self.df[numeric_cols[0]].values
            self.y = self.df[numeric_cols[1]].values
            print(f"\nUsing columns: X={numeric_cols[0]}, Y={numeric_cols[1]}")
    
    def descriptive_statistics(self):
        """1. Compute descriptive statistics for X and Y"""
        print("\n" + "="*80)
        print("1. DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # For X (Experience)
        print("\n📊 Statistics for X (Experience):")
        print("-" * 50)
        x_stats = {
            'Count': len(self.x),
            'Mean': np.mean(self.x),
            'Median': np.median(self.x),
            'Mode': stats.mode(self.x, keepdims=True)[0][0],
            'Standard Deviation': np.std(self.x, ddof=1),
            'Variance': np.var(self.x, ddof=1),
            'Minimum': np.min(self.x),
            'Maximum': np.max(self.x),
            'Range': np.max(self.x) - np.min(self.x),
            'Q1 (25th percentile)': np.percentile(self.x, 25),
            'Q2 (50th percentile)': np.percentile(self.x, 50),
            'Q3 (75th percentile)': np.percentile(self.x, 75),
            'IQR': np.percentile(self.x, 75) - np.percentile(self.x, 25),
            'Skewness': stats.skew(self.x),
            'Kurtosis': stats.kurtosis(self.x),
            'Coefficient of Variation': (np.std(self.x, ddof=1) / np.mean(self.x)) * 100
        }
        
        for key, value in x_stats.items():
            print(f"{key:.<30} {value:.4f}")
        
        # For Y (Salary)
        print("\n📊 Statistics for Y (Salary):")
        print("-" * 50)
        y_stats = {
            'Count': len(self.y),
            'Mean': np.mean(self.y),
            'Median': np.median(self.y),
            'Mode': stats.mode(self.y, keepdims=True)[0][0],
            'Standard Deviation': np.std(self.y, ddof=1),
            'Variance': np.var(self.y, ddof=1),
            'Minimum': np.min(self.y),
            'Maximum': np.max(self.y),
            'Range': np.max(self.y) - np.min(self.y),
            'Q1 (25th percentile)': np.percentile(self.y, 25),
            'Q2 (50th percentile)': np.percentile(self.y, 50),
            'Q3 (75th percentile)': np.percentile(self.y, 75),
            'IQR': np.percentile(self.y, 75) - np.percentile(self.y, 25),
            'Skewness': stats.skew(self.y),
            'Kurtosis': stats.kurtosis(self.y),
            'Coefficient of Variation': (np.std(self.y, ddof=1) / np.mean(self.y)) * 100
        }
        
        for key, value in y_stats.items():
            print(f"{key:.<30} {value:.4f}")
        
        # Correlation
        correlation = np.corrcoef(self.x, self.y)[0, 1]
        print("\n📈 Correlation Analysis:")
        print("-" * 50)
        print(f"Pearson Correlation Coefficient: {correlation:.4f}")
        
        # Visualization
        self._plot_descriptive_stats()
        
        return x_stats, y_stats
    
    def _plot_descriptive_stats(self):
        """Create visualizations for descriptive statistics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Descriptive Statistics Visualization', fontsize=16, fontweight='bold')
        
        # Histogram for X
        axes[0, 0].hist(self.x, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(self.x), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.x):.2f}')
        axes[0, 0].axvline(np.median(self.x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(self.x):.2f}')
        axes[0, 0].set_xlabel('Experience (Years)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Experience')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram for Y
        axes[0, 1].hist(self.y, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(self.y), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.y):.2f}')
        axes[0, 1].axvline(np.median(self.y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(self.y):.2f}')
        axes[0, 1].set_xlabel('Salary')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Salary')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot for X
        axes[0, 2].boxplot(self.x, vert=True)
        axes[0, 2].set_ylabel('Experience (Years)')
        axes[0, 2].set_title('Box Plot of Experience')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Box plot for Y
        axes[1, 0].boxplot(self.y, vert=True)
        axes[1, 0].set_ylabel('Salary')
        axes[1, 0].set_title('Box Plot of Salary')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1, 1].scatter(self.x, self.y, alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Experience (Years)')
        axes[1, 1].set_ylabel('Salary')
        axes[1, 1].set_title(f'Experience vs Salary (r={np.corrcoef(self.x, self.y)[0,1]:.3f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Q-Q plots
        stats.probplot(self.x, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot for Experience')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/descriptive_statistics.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved as 'output/descriptive_statistics.png'")
        plt.close()
    
    def confidence_intervals(self, confidence_level=0.95):
        """2. Calculate confidence intervals for mean X and mean Y"""
        print("\n" + "="*80)
        print(f"2. CONFIDENCE INTERVALS (Confidence Level: {confidence_level*100}%)")
        print("="*80)
        
        alpha = 1 - confidence_level
        
        # CI for mean X
        print("\n📊 Confidence Interval for Mean X (Experience):")
        print("-" * 50)
        x_mean = np.mean(self.x)
        x_std = np.std(self.x, ddof=1)
        x_n = len(self.x)
        x_se = x_std / np.sqrt(x_n)
        
        # Using t-distribution
        x_t_critical = stats.t.ppf(1 - alpha/2, df=x_n-1)
        x_ci_lower = x_mean - x_t_critical * x_se
        x_ci_upper = x_mean + x_t_critical * x_se
        x_margin_error = x_t_critical * x_se
        
        print(f"Sample size (n)................ {x_n}")
        print(f"Sample mean (x̄)............... {x_mean:.4f}")
        print(f"Sample std dev (s)............. {x_std:.4f}")
        print(f"Standard error (SE)............ {x_se:.4f}")
        print(f"Degrees of freedom............. {x_n-1}")
        print(f"t-critical value............... {x_t_critical:.4f}")
        print(f"Margin of error................ {x_margin_error:.4f}")
        print(f"\n✨ {confidence_level*100}% CI: [{x_ci_lower:.4f}, {x_ci_upper:.4f}]")
        print(f"\nInterpretation: We are {confidence_level*100}% confident that the true mean")
        print(f"experience lies between {x_ci_lower:.4f} and {x_ci_upper:.4f} years.")
        
        # CI for mean Y
        print("\n📊 Confidence Interval for Mean Y (Salary):")
        print("-" * 50)
        y_mean = np.mean(self.y)
        y_std = np.std(self.y, ddof=1)
        y_n = len(self.y)
        y_se = y_std / np.sqrt(y_n)
        
        y_t_critical = stats.t.ppf(1 - alpha/2, df=y_n-1)
        y_ci_lower = y_mean - y_t_critical * y_se
        y_ci_upper = y_mean + y_t_critical * y_se
        y_margin_error = y_t_critical * y_se
        
        print(f"Sample size (n)................ {y_n}")
        print(f"Sample mean (ȳ)............... {y_mean:.4f}")
        print(f"Sample std dev (s)............. {y_std:.4f}")
        print(f"Standard error (SE)............ {y_se:.4f}")
        print(f"Degrees of freedom............. {y_n-1}")
        print(f"t-critical value............... {y_t_critical:.4f}")
        print(f"Margin of error................ {y_margin_error:.4f}")
        print(f"\n✨ {confidence_level*100}% CI: [{y_ci_lower:.4f}, {y_ci_upper:.4f}]")
        print(f"\nInterpretation: We are {confidence_level*100}% confident that the true mean")
        print(f"salary lies between ${y_ci_lower:.2f} and ${y_ci_upper:.2f}.")
        
        # Visualization
        self._plot_confidence_intervals(x_mean, x_ci_lower, x_ci_upper, 
                                       y_mean, y_ci_lower, y_ci_upper, confidence_level)
        
        return {
            'X': {'mean': x_mean, 'ci_lower': x_ci_lower, 'ci_upper': x_ci_upper, 'margin_error': x_margin_error},
            'Y': {'mean': y_mean, 'ci_lower': y_ci_lower, 'ci_upper': y_ci_upper, 'margin_error': y_margin_error}
        }
    
    def _plot_confidence_intervals(self, x_mean, x_ci_lower, x_ci_upper, 
                                   y_mean, y_ci_lower, y_ci_upper, confidence_level):
        """Visualize confidence intervals"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{confidence_level*100}% Confidence Intervals for Means', 
                    fontsize=16, fontweight='bold')
        
        # CI for X
        axes[0].errorbar(['Experience'], [x_mean], 
                        yerr=[[x_mean - x_ci_lower], [x_ci_upper - x_mean]], 
                        fmt='o', markersize=10, capsize=10, capthick=2, 
                        color='blue', ecolor='blue', linewidth=2)
        axes[0].axhline(y=x_mean, color='red', linestyle='--', alpha=0.5, label=f'Mean: {x_mean:.2f}')
        axes[0].fill_between([0, 1], x_ci_lower, x_ci_upper, alpha=0.2, color='blue')
        axes[0].set_ylabel('Years')
        axes[0].set_title('CI for Mean Experience')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-0.5, 0.5)
        
        # CI for Y
        axes[1].errorbar(['Salary'], [y_mean], 
                        yerr=[[y_mean - y_ci_lower], [y_ci_upper - y_mean]], 
                        fmt='o', markersize=10, capsize=10, capthick=2, 
                        color='green', ecolor='green', linewidth=2)
        axes[1].axhline(y=y_mean, color='red', linestyle='--', alpha=0.5, label=f'Mean: ${y_mean:.2f}')
        axes[1].fill_between([0, 1], y_ci_lower, y_ci_upper, alpha=0.2, color='green')
        axes[1].set_ylabel('Salary ($)')
        axes[1].set_title('CI for Mean Salary')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(-0.5, 0.5)
        
        plt.tight_layout()
        plt.savefig('output/confidence_intervals.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved as 'output/confidence_intervals.png'")
        plt.close()
    
    def hypothesis_tests(self, mu0_x=None, mu0_y=None, alpha=0.05):
        """3. Perform hypothesis tests for mean X and mean Y"""
        print("\n" + "="*80)
        print(f"3. HYPOTHESIS TESTS FOR MEANS (Significance Level: α={alpha})")
        print("="*80)
        
        # Test for mean X
        print("\n📊 One-Sample t-Test for Mean X (Experience):")
        print("-" * 50)
        
        if mu0_x is None:
            mu0_x = 5.0  # Default hypothesized mean
            print(f"Using default hypothesized mean: μ₀ = {mu0_x}")
        
        print(f"\nHypotheses:")
        print(f"  H₀: μ = {mu0_x} (null hypothesis)")
        print(f"  H₁: μ ≠ {mu0_x} (alternative hypothesis - two-tailed)")
        
        x_mean = np.mean(self.x)
        x_std = np.std(self.x, ddof=1)
        x_n = len(self.x)
        x_se = x_std / np.sqrt(x_n)
        
        # Calculate t-statistic
        t_stat_x = (x_mean - mu0_x) / x_se
        df_x = x_n - 1
        
        # Calculate p-value (two-tailed)
        p_value_x = 2 * (1 - stats.t.cdf(abs(t_stat_x), df=df_x))
        
        # Critical value
        t_critical_x = stats.t.ppf(1 - alpha/2, df=df_x)
        
        print(f"\nTest Statistics:")
        print(f"  Sample mean (x̄).............. {x_mean:.4f}")
        print(f"  Hypothesized mean (μ₀)........ {mu0_x:.4f}")
        print(f"  Standard error (SE)........... {x_se:.4f}")
        print(f"  t-statistic................... {t_stat_x:.4f}")
        print(f"  Degrees of freedom............ {df_x}")
        print(f"  Critical value (±)............ {t_critical_x:.4f}")
        print(f"  p-value....................... {p_value_x:.4f}")
        
        print(f"\nDecision:")
        if p_value_x < alpha:
            print(f"  ✅ Reject H₀ (p-value {p_value_x:.4f} < α {alpha})")
            print(f"  Conclusion: There is significant evidence that the mean experience")
            print(f"              differs from {mu0_x} years.")
        else:
            print(f"  ❌ Fail to reject H₀ (p-value {p_value_x:.4f} ≥ α {alpha})")
            print(f"  Conclusion: There is insufficient evidence that the mean experience")
            print(f"              differs from {mu0_x} years.")
        
        # Test for mean Y
        print("\n📊 One-Sample t-Test for Mean Y (Salary):")
        print("-" * 50)
        
        if mu0_y is None:
            mu0_y = 60000.0  # Default hypothesized mean
            print(f"Using default hypothesized mean: μ₀ = ${mu0_y}")
        
        print(f"\nHypotheses:")
        print(f"  H₀: μ = {mu0_y} (null hypothesis)")
        print(f"  H₁: μ ≠ {mu0_y} (alternative hypothesis - two-tailed)")
        
        y_mean = np.mean(self.y)
        y_std = np.std(self.y, ddof=1)
        y_n = len(self.y)
        y_se = y_std / np.sqrt(y_n)
        
        t_stat_y = (y_mean - mu0_y) / y_se
        df_y = y_n - 1
        p_value_y = 2 * (1 - stats.t.cdf(abs(t_stat_y), df=df_y))
        t_critical_y = stats.t.ppf(1 - alpha/2, df=df_y)
        
        print(f"\nTest Statistics:")
        print(f"  Sample mean (ȳ)............... ${y_mean:.2f}")
        print(f"  Hypothesized mean (μ₀)........ ${mu0_y:.2f}")
        print(f"  Standard error (SE)........... {y_se:.4f}")
        print(f"  t-statistic................... {t_stat_y:.4f}")
        print(f"  Degrees of freedom............ {df_y}")
        print(f"  Critical value (±)............ {t_critical_y:.4f}")
        print(f"  p-value....................... {p_value_y:.4f}")
        
        print(f"\nDecision:")
        if p_value_y < alpha:
            print(f"  ✅ Reject H₀ (p-value {p_value_y:.4f} < α {alpha})")
            print(f"  Conclusion: There is significant evidence that the mean salary")
            print(f"              differs from ${mu0_y:.2f}.")
        else:
            print(f"  ❌ Fail to reject H₀ (p-value {p_value_y:.4f} ≥ α {alpha})")
            print(f"  Conclusion: There is insufficient evidence that the mean salary")
            print(f"              differs from ${mu0_y:.2f}.")
        
        # Additional tests
        print("\n📊 Additional Tests:")
        print("-" * 50)
        
        # Normality tests
        _, p_shapiro_x = stats.shapiro(self.x)
        _, p_shapiro_y = stats.shapiro(self.y)
        
        print("\nNormality Tests (Shapiro-Wilk):")
        print(f"  Experience: p-value = {p_shapiro_x:.4f}")
        if p_shapiro_x > alpha:
            print(f"    ✅ Data appears normally distributed")
        else:
            print(f"    ⚠️  Data may not be normally distributed")
        
        print(f"  Salary: p-value = {p_shapiro_y:.4f}")
        if p_shapiro_y > alpha:
            print(f"    ✅ Data appears normally distributed")
        else:
            print(f"    ⚠️  Data may not be normally distributed")
        
        return {
            'X': {'t_stat': t_stat_x, 'p_value': p_value_x, 'reject_h0': p_value_x < alpha},
            'Y': {'t_stat': t_stat_y, 'p_value': p_value_y, 'reject_h0': p_value_y < alpha}
        }
    
    def regression_analysis(self):
        """4. Perform regression analysis"""
        print("\n" + "="*80)
        print("4. REGRESSION ANALYSIS")
        print("="*80)
        
        # Prepare data
        X = self.x.reshape(-1, 1)
        y = self.y
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Model parameters
        slope = model.coef_[0]
        intercept = model.intercept_
        
        print("\n📈 Simple Linear Regression Model:")
        print("-" * 50)
        print(f"\nRegression Equation:")
        print(f"  Salary = {intercept:.4f} + {slope:.4f} × Experience")
        print(f"  ŷ = {intercept:.4f} + {slope:.4f}x")
        
        # Model evaluation
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - 2)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        print(f"\nModel Performance Metrics:")
        print(f"  R² (Coefficient of Determination).... {r2:.4f}")
        print(f"  Adjusted R²........................... {adj_r2:.4f}")
        print(f"  Mean Squared Error (MSE).............. {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE)........ {rmse:.4f}")
        print(f"  Mean Absolute Error (MAE)............. {mae:.4f}")
        
        print(f"\nInterpretation:")
        print(f"  - For each additional year of experience, salary increases by ${slope:.2f}")
        print(f"  - The model explains {r2*100:.2f}% of the variance in salary")
        print(f"  - Starting salary (0 years exp) is estimated at ${intercept:.2f}")
        
        # Statistical significance of regression
        print("\n📊 Statistical Significance:")
        print("-" * 50)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Standard error of the regression
        n = len(y)
        sse = np.sum(residuals**2)
        se_regression = np.sqrt(sse / (n - 2))
        
        # Standard error of the slope
        x_mean = np.mean(self.x)
        sxx = np.sum((self.x - x_mean)**2)
        se_slope = se_regression / np.sqrt(sxx)
        
        # t-statistic for slope
        t_stat_slope = slope / se_slope
        p_value_slope = 2 * (1 - stats.t.cdf(abs(t_stat_slope), df=n-2))
        
        print(f"\nRegression Coefficients:")
        print(f"  Intercept (β₀):")
        print(f"    Estimate......................... {intercept:.4f}")
        print(f"  Slope (β₁):")
        print(f"    Estimate......................... {slope:.4f}")
        print(f"    Standard Error................... {se_slope:.4f}")
        print(f"    t-statistic...................... {t_stat_slope:.4f}")
        print(f"    p-value.......................... {p_value_slope:.6f}")
        
        if p_value_slope < 0.05:
            print(f"\n  ✅ The slope is statistically significant (p < 0.05)")
            print(f"     Experience has a significant effect on Salary")
        else:
            print(f"\n  ❌ The slope is not statistically significant (p ≥ 0.05)")
        
        # ANOVA for regression
        print("\n📊 ANOVA Table:")
        print("-" * 50)
        
        y_mean = np.mean(y)
        sst = np.sum((y - y_mean)**2)  # Total sum of squares
        ssr = np.sum((y_pred - y_mean)**2)  # Regression sum of squares
        sse = np.sum((y - y_pred)**2)  # Error sum of squares
        
        df_regression = 1
        df_residual = n - 2
        df_total = n - 1
        
        msr = ssr / df_regression
        mse = sse / df_residual
        
        f_stat = msr / mse
        p_value_f = 1 - stats.f.cdf(f_stat, df_regression, df_residual)
        
        print(f"\n{'Source':<15} {'SS':<15} {'df':<10} {'MS':<15} {'F':<15} {'p-value'}")
        print("-" * 80)
        print(f"{'Regression':<15} {ssr:<15.4f} {df_regression:<10} {msr:<15.4f} {f_stat:<15.4f} {p_value_f:.6f}")
        print(f"{'Residual':<15} {sse:<15.4f} {df_residual:<10} {mse:<15.4f}")
        print(f"{'Total':<15} {sst:<15.4f} {df_total:<10}")
        
        # Residual analysis
        print("\n📊 Residual Analysis:")
        print("-" * 50)
        print(f"  Mean of residuals.................... {np.mean(residuals):.6f}")
        print(f"  Std dev of residuals................. {np.std(residuals):.4f}")
        print(f"  Min residual......................... {np.min(residuals):.4f}")
        print(f"  Max residual......................... {np.max(residuals):.4f}")
        
        # Durbin-Watson test for autocorrelation
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        print(f"  Durbin-Watson statistic.............. {dw:.4f}")
        if 1.5 < dw < 2.5:
            print(f"    ✅ No significant autocorrelation detected")
        else:
            print(f"    ⚠️  Possible autocorrelation in residuals")
        
        # Visualization
        self._plot_regression(X, y, y_pred, residuals, slope, intercept, r2)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'adjusted_r2': adj_r2,
            'rmse': rmse,
            'p_value': p_value_slope,
            'model': model
        }
    
    def _plot_regression(self, X, y, y_pred, residuals, slope, intercept, r2):
        """Create comprehensive regression visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Regression Analysis Visualization', fontsize=16, fontweight='bold')
        
        # 1. Regression line plot
        axes[0, 0].scatter(X, y, alpha=0.6, label='Actual data', color='blue')
        axes[0, 0].plot(X, y_pred, 'r-', linewidth=2, label=f'Regression line\nŷ = {intercept:.2f} + {slope:.2f}x')
        axes[0, 0].set_xlabel('Experience (Years)')
        axes[0, 0].set_ylabel('Salary ($)')
        axes[0, 0].set_title(f'Linear Regression (R² = {r2:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Fitted values
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='purple')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Fitted Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Normal Q-Q Plot of Residuals')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Histogram of residuals
        axes[1, 0].hist(residuals, bins=20, color='green', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(residuals):.2e}')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Scale-Location plot
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, color='orange')
        axes[1, 1].set_xlabel('Fitted Values')
        axes[1, 1].set_ylabel('√|Standardized Residuals|')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Residuals vs Order
        axes[1, 2].plot(range(len(residuals)), residuals, 'o-', alpha=0.6, color='brown')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_xlabel('Observation Order')
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title('Residuals vs Order')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/regression_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved as 'output/regression_analysis.png'")
        plt.close()

def main():
    """Main function to run the complete analysis"""
    print("\n" + "="*80)
    print(" " * 20 + "EXPERIENCE-SALARY STATISTICAL ANALYSIS")
    print(" " * 25 + "Data Science Project - HCMIU")
    print("="*80)
    
    # Load data (adjust path as needed)
    data_path = 'Salary_Data.csv'  # Change to your actual file path
    
    try:
        # Initialize analysis
        analysis = ExperienceSalaryAnalysis(data_path)
        
        # 1. Descriptive Statistics
        analysis.descriptive_statistics()
        
        # 2. Confidence Intervals
        analysis.confidence_intervals(confidence_level=0.95)
        
        # 3. Hypothesis Tests
        # You can customize the hypothesized means
        analysis.hypothesis_tests(mu0_x=5.0, mu0_y=60000.0, alpha=0.05)
        
        # 4. Regression Analysis
        analysis.regression_analysis()
        
        print("\n" + "="*80)
        print(" " * 30 + "ANALYSIS COMPLETE!")
        print("="*80)
        print("\n📁 Output files generated:")
        print("  - output/descriptive_statistics.png")
        print("  - output/confidence_intervals.png")
        print("  - output/regression_analysis.png")
        print("\n✨ All statistical analyses have been completed successfully!\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{data_path}' not found!")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/saquib7hussain/experience-salary-dataset/data")
        print(f"\nAnd save it as '{data_path}' in the same directory as this script.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
