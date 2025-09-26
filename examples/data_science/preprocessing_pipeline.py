#!/usr/bin/env python3
"""
Preprocessing Pipeline Examples - Refunc Data Science

This example demonstrates comprehensive data preprocessing pipelines including
data transformation, cleaning, feature engineering, and preprocessing workflows
for ML-ready data preparation.

Key Features Demonstrated:
- Transformation pipelines and composition
- Missing value imputation strategies
- Data scaling and normalization
- Categorical encoding techniques
- Feature engineering automation
- Pipeline result tracking and validation
"""

import os
import sys
import time
import random
import math
from typing import List, Dict, Any, Optional, Tuple, Union

# Handle missing dependencies gracefully
try:
    from refunc.data_science import (
        TransformationPipeline, BaseTransformer,
        MissingValueImputer, DataScaler, OutlierRemover,
        CategoricalEncoder, CustomTransformer,
        create_basic_pipeline, create_robust_pipeline,
        apply_quick_preprocessing, TransformationType
    )
    # Try to import pandas for examples
    try:
        import pandas as pd
        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False
        print("📊 Pandas not available - will use mock data structures")
    
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False
    PANDAS_AVAILABLE = False


def create_sample_dataset():
    """Create sample dataset with various data quality issues."""
    if PANDAS_AVAILABLE:
        # Create realistic dataset with multiple data types and issues
        n_samples = 500
        
        data = {
            'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
            'age': [random.randint(18, 80) if random.random() > 0.05 else None for _ in range(n_samples)],
            'income': [random.uniform(20000, 150000) if random.random() > 0.08 else None for _ in range(n_samples)],
            'credit_score': [random.randint(300, 850) if random.random() > 0.06 else None for _ in range(n_samples)],
            'account_balance': [random.uniform(-5000, 100000) for _ in range(n_samples)],
            'years_customer': [random.randint(0, 25) for _ in range(n_samples)],
            'region': [random.choice(['North', 'South', 'East', 'West', 'Unknown']) for _ in range(n_samples)],
            'product_type': [random.choice(['Basic', 'Premium', 'Enterprise', 'Trial']) for _ in range(n_samples)],
            'is_active': [random.choice([True, False]) for _ in range(n_samples)],
            'last_login_days': [random.randint(0, 365) if random.random() > 0.1 else None for _ in range(n_samples)],
            'transaction_count': [random.randint(0, 500) for _ in range(n_samples)],
            'support_tickets': [random.randint(0, 20) for _ in range(n_samples)]
        }
        
        # Add some extreme outliers
        outlier_indices = random.sample(range(n_samples), 15)
        for idx in outlier_indices:
            if random.random() < 0.3:
                data['income'][idx] = random.choice([500000, 1000000, -10000])
            if random.random() < 0.3:
                data['account_balance'][idx] = random.choice([500000, -50000])
            if random.random() < 0.3:
                data['age'][idx] = random.choice([150, 200, -5])
        
        return pd.DataFrame(data)
    else:
        # Mock dataset structure
        return {
            'data': [
                {'customer_id': 'CUST_001', 'age': 25, 'income': 50000, 'region': 'North', 'is_active': True},
                {'customer_id': 'CUST_002', 'age': None, 'income': 75000, 'region': 'South', 'is_active': False},
                {'customer_id': 'CUST_003', 'age': 35, 'income': None, 'region': 'East', 'is_active': True},
                {'customer_id': 'CUST_004', 'age': 45, 'income': 120000, 'region': 'Unknown', 'is_active': True}
            ],
            'columns': ['customer_id', 'age', 'income', 'region', 'is_active'],
            'shape': (4, 5)
        }


def basic_transformation_examples():
    """Demonstrate basic data transformation operations."""
    print("🔧 Basic Data Transformations")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic transformation examples:
from refunc.data_science import TransformationPipeline, MissingValueImputer

# Create transformation pipeline
pipeline = TransformationPipeline()

# Add missing value imputation
imputer = MissingValueImputer(strategy='mean', columns=['age', 'income'])
pipeline.add_step('impute_missing', imputer)

# Add data scaling
scaler = DataScaler(method='standard', columns=['age', 'income', 'balance'])
pipeline.add_step('scale_data', scaler)

# Apply pipeline
result = pipeline.fit_transform(df)
print(f"Transformed data shape: {result.data.shape}")
print(f"Pipeline steps: {len(result.steps_applied)}")
        """)
        return
    
    print("⚙️ Testing basic transformations:")
    
    sample_data = create_sample_dataset()
    
    if PANDAS_AVAILABLE:
        print(f"   📊 Original dataset: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)}")
        
        # Check data quality before transformation
        print(f"\n   📋 Data quality assessment:")
        missing_counts = sample_data.isnull().sum()
        total_missing = missing_counts.sum()
        print(f"     Total missing values: {total_missing}")
        
        # Show missing values by column
        for column in ['age', 'income', 'credit_score', 'last_login_days']:
            if column in sample_data.columns:
                missing_count = missing_counts[column]
                missing_pct = (missing_count / len(sample_data)) * 100
                print(f"     {column}: {missing_count} missing ({missing_pct:.1f}%)")
        
        # Manual missing value imputation
        print(f"\n   🔄 Applying missing value imputation:")
        
        # Numeric imputation (mean/median)
        numeric_columns = ['age', 'income', 'credit_score', 'last_login_days']
        processed_data = sample_data.copy()
        
        imputation_stats = {}
        for column in numeric_columns:
            if column in processed_data.columns and processed_data[column].isnull().any():
                original_missing = processed_data[column].isnull().sum()
                
                # Use median for age and credit_score, mean for others
                if column in ['age', 'credit_score']:
                    fill_value = processed_data[column].median()
                    method = 'median'
                else:
                    fill_value = processed_data[column].mean()
                    method = 'mean'
                
                processed_data[column] = processed_data[column].fillna(fill_value)
                
                imputation_stats[column] = {
                    'method': method,
                    'fill_value': fill_value,
                    'imputed_count': original_missing
                }
                
                print(f"     {column}: {original_missing} values imputed with {method} ({fill_value:.1f})")
        
        # Data type optimization
        print(f"\n   📐 Optimizing data types:")
        
        # Convert boolean columns
        if 'is_active' in processed_data.columns:
            processed_data['is_active'] = processed_data['is_active'].astype('bool')
            print(f"     is_active: converted to boolean")
        
        # Convert categorical columns
        categorical_columns = ['region', 'product_type']
        for column in categorical_columns:
            if column in processed_data.columns:
                original_type = processed_data[column].dtype
                processed_data[column] = processed_data[column].astype('category') 
                print(f"     {column}: {original_type} -> category")
        
        # Check memory usage improvement
        original_memory = sample_data.memory_usage(deep=True).sum() / 1024**2  # MB
        optimized_memory = processed_data.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
        
        print(f"\n   💾 Memory optimization:")
        print(f"     Original: {original_memory:.2f} MB")
        print(f"     Optimized: {optimized_memory:.2f} MB")
        print(f"     Reduction: {memory_reduction:.1f}%")
        
        print(f"\n   ✅ Basic transformation completed:")
        print(f"     Final shape: {processed_data.shape}")
        print(f"     Remaining missing values: {processed_data.isnull().sum().sum()}")
        
    else:
        print("   🎭 Mock transformation (pandas not available):")
        print("   Original dataset: (4, 5) shape")
        print("   Data quality assessment:")
        print("     Total missing values: 2")
        print("     age: 1 missing (25.0%)")
        print("     income: 1 missing (25.0%)")
        print("   Applying missing value imputation:")
        print("     age: 1 values imputed with median (35.0)")
        print("     income: 1 values imputed with mean (81666.7)")
        print("   Final shape: (4, 5)")
        print("   Remaining missing values: 0")


def feature_engineering_examples():
    """Demonstrate feature engineering and creation."""
    print("\n🏗️ Feature Engineering")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Feature engineering examples:
from refunc.data_science import FeatureEngineer, TransformationPipeline

# Create feature engineering pipeline
engineer = FeatureEngineer()

# Add derived features
engineer.add_ratio_feature('debt_to_income', 'debt', 'income')
engineer.add_binning_feature('age_group', 'age', bins=[0, 25, 45, 65, 100])
engineer.add_interaction_feature('age_income', ['age', 'income'])

# Add time-based features
engineer.add_date_features('last_login', ['year', 'month', 'dayofweek'])

# Apply feature engineering
result = engineer.fit_transform(df)
print(f"Original features: {len(df.columns)}")
print(f"Engineered features: {len(result.data.columns)}")
        """)
        return
    
    print("🔬 Testing feature engineering:")
    
    sample_data = create_sample_dataset()
    
    if PANDAS_AVAILABLE:
        # Start with preprocessed data (no missing values)
        processed_data = sample_data.copy()
        
        # Fill missing values first
        numeric_columns = ['age', 'income', 'credit_score', 'last_login_days']
        for column in numeric_columns:
            if column in processed_data.columns and processed_data[column].isnull().any():
                if column in ['age', 'credit_score']:
                    processed_data[column] = processed_data[column].fillna(processed_data[column].median())
                else:
                    processed_data[column] = processed_data[column].fillna(processed_data[column].mean())
        
        print(f"   📊 Starting with clean data: {processed_data.shape}")
        original_columns = len(processed_data.columns)
        
        # Feature engineering
        engineered_data = processed_data.copy()
        
        # 1. Ratio features
        print(f"\n   📐 Creating ratio features:")
        if 'income' in engineered_data.columns and 'account_balance' in engineered_data.columns:
            # Balance to income ratio
            engineered_data['balance_to_income_ratio'] = (
                engineered_data['account_balance'] / engineered_data['income']
            )
            print(f"     balance_to_income_ratio: min={engineered_data['balance_to_income_ratio'].min():.2f}, "
                  f"max={engineered_data['balance_to_income_ratio'].max():.2f}")
        
        if 'transaction_count' in engineered_data.columns and 'years_customer' in engineered_data.columns:
            # Transaction rate per year
            engineered_data['transactions_per_year'] = (
                engineered_data['transaction_count'] / (engineered_data['years_customer'] + 1)  # +1 to avoid division by zero
            )
            print(f"     transactions_per_year: mean={engineered_data['transactions_per_year'].mean():.1f}")
        
        # 2. Binning features
        print(f"\n   📊 Creating binned features:")
        if 'age' in engineered_data.columns:
            age_bins = [0, 25, 35, 50, 65, 100]
            age_labels = ['Young', 'Adult', 'Middle-aged', 'Senior', 'Elder']
            engineered_data['age_group'] = pd.cut(
                engineered_data['age'], 
                bins=age_bins, 
                labels=age_labels, 
                include_lowest=True
            )
            age_group_counts = engineered_data['age_group'].value_counts()
            print(f"     age_group: {len(age_group_counts)} categories")
            for group, count in age_group_counts.head(3).items():
                print(f"       {group}: {count} samples")
        
        if 'income' in engineered_data.columns:
            income_bins = [0, 30000, 60000, 100000, float('inf')]
            income_labels = ['Low', 'Medium', 'High', 'Very High']
            engineered_data['income_bracket'] = pd.cut(
                engineered_data['income'],
                bins=income_bins,
                labels=income_labels,
                include_lowest=True
            )
            income_bracket_counts = engineered_data['income_bracket'].value_counts()
            print(f"     income_bracket: {len(income_bracket_counts)} categories")
        
        # 3. Interaction features
        print(f"\n   🔗 Creating interaction features:")
        if 'age' in engineered_data.columns and 'income' in engineered_data.columns:
            # Age-income interaction
            engineered_data['age_income_interaction'] = (
                engineered_data['age'] * engineered_data['income'] / 1000  # Scale down
            )
            print(f"     age_income_interaction: mean={engineered_data['age_income_interaction'].mean():.0f}")
        
        if 'years_customer' in engineered_data.columns and 'transaction_count' in engineered_data.columns:
            # Customer loyalty score
            engineered_data['loyalty_score'] = (
                engineered_data['years_customer'] * 0.3 + 
                engineered_data['transaction_count'] * 0.01
            )
            print(f"     loyalty_score: mean={engineered_data['loyalty_score'].mean():.2f}")
        
        # 4. Aggregated features
        print(f"\n   📈 Creating aggregated features:")
        
        # Customer activity level
        activity_features = ['transaction_count', 'last_login_days', 'support_tickets']
        activity_scores = []
        
        for idx, row in engineered_data.iterrows():
            score = 0
            if 'transaction_count' in row and pd.notna(row['transaction_count']):
                score += min(row['transaction_count'] / 100, 5)  # Max 5 points
            if 'last_login_days' in row and pd.notna(row['last_login_days']):
                score += max(5 - row['last_login_days'] / 30, 0)  # Fewer days = higher score
            if 'support_tickets' in row and pd.notna(row['support_tickets']):
                score -= row['support_tickets'] * 0.1  # More tickets = lower score
            
            activity_scores.append(max(score, 0))
        
        engineered_data['activity_score'] = activity_scores
        print(f"     activity_score: mean={engineered_data['activity_score'].mean():.2f}, "
              f"std={engineered_data['activity_score'].std():.2f}")
        
        # 5. Boolean flags
        print(f"\n   🏁 Creating boolean flag features:")
        
        # High value customer flag
        if 'income' in engineered_data.columns:
            high_income_threshold = engineered_data['income'].quantile(0.8)
            engineered_data['is_high_income'] = engineered_data['income'] > high_income_threshold
            high_income_count = engineered_data['is_high_income'].sum()
            print(f"     is_high_income: {high_income_count} customers ({high_income_count/len(engineered_data)*100:.1f}%)")
        
        # Long-term customer flag
        if 'years_customer' in engineered_data.columns:
            engineered_data['is_long_term'] = engineered_data['years_customer'] >= 5
            long_term_count = engineered_data['is_long_term'].sum()
            print(f"     is_long_term: {long_term_count} customers ({long_term_count/len(engineered_data)*100:.1f}%)")
        
        # At-risk customer flag (high support tickets, low activity)
        if 'support_tickets' in engineered_data.columns and 'last_login_days' in engineered_data.columns:
            engineered_data['is_at_risk'] = (
                (engineered_data['support_tickets'] > engineered_data['support_tickets'].quantile(0.75)) |
                (engineered_data['last_login_days'] > 90)
            )
            at_risk_count = engineered_data['is_at_risk'].sum()
            print(f"     is_at_risk: {at_risk_count} customers ({at_risk_count/len(engineered_data)*100:.1f}%)")
        
        # Summary
        final_columns = len(engineered_data.columns)
        new_features = final_columns - original_columns
        
        print(f"\n   ✅ Feature engineering completed:")
        print(f"     Original features: {original_columns}")
        print(f"     New features: {new_features}")
        print(f"     Total features: {final_columns}")
        print(f"     Feature increase: {(new_features/original_columns)*100:.1f}%")
        
        # Show sample of new features
        new_feature_columns = [col for col in engineered_data.columns if col not in processed_data.columns]
        print(f"     New feature samples: {new_feature_columns[:5]}")
        
    else:
        print("   🎭 Mock feature engineering (pandas not available):")
        print("   Starting with clean data: (4, 5) shape")
        print("   Creating ratio features:")
        print("     balance_to_income_ratio: min=-0.83, max=2.40")
        print("     transactions_per_year: mean=45.2")
        print("   Creating binned features:")
        print("     age_group: 4 categories (Young, Adult, Middle-aged, Senior)")
        print("     income_bracket: 3 categories (Medium, High, Very High)")
        print("   Creating interaction features:")
        print("     age_income_interaction: mean=2840")
        print("     loyalty_score: mean=8.45")
        print("   Feature engineering completed:")
        print("     Original features: 5")
        print("     New features: 8")
        print("     Total features: 13")


def scaling_normalization_examples():
    """Demonstrate data scaling and normalization techniques."""
    print("\n📏 Data Scaling and Normalization")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Scaling and normalization examples:
from refunc.data_science import DataScaler, TransformationPipeline

# Standard scaling (z-score normalization)
standard_scaler = DataScaler(method='standard', columns=['age', 'income'])
scaled_data = standard_scaler.fit_transform(df)

# Min-max scaling
minmax_scaler = DataScaler(method='minmax', columns=['balance', 'score'])
normalized_data = minmax_scaler.fit_transform(df)

# Robust scaling (median and IQR)
robust_scaler = DataScaler(method='robust', columns=['transaction_count'])
robust_data = robust_scaler.fit_transform(df)

# Custom scaling with pipeline
pipeline = TransformationPipeline()
pipeline.add_step('scale_numeric', DataScaler(method='standard'))
pipeline.add_step('normalize_skewed', DataScaler(method='log', columns=['income']))
        """)
        return
    
    print("📊 Testing scaling and normalization:")
    
    sample_data = create_sample_dataset()
    
    if PANDAS_AVAILABLE:
        # Prepare clean numeric data
        processed_data = sample_data.copy()
        
        # Fill missing values first
        numeric_columns = ['age', 'income', 'credit_score', 'account_balance', 'years_customer', 
                          'last_login_days', 'transaction_count', 'support_tickets']
        
        for column in numeric_columns:
            if column in processed_data.columns and processed_data[column].isnull().any():
                fill_value = processed_data[column].median()
                processed_data[column] = processed_data[column].fillna(fill_value)
        
        print(f"   📊 Analyzing data distributions:")
        
        # Analyze distributions before scaling
        distribution_stats = {}
        scaling_candidates = ['age', 'income', 'credit_score', 'account_balance', 'transaction_count']
        
        for column in scaling_candidates:
            if column in processed_data.columns:
                col_data = processed_data[column]
                stats = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'skewness': col_data.skew() if hasattr(col_data, 'skew') else 0
                }
                distribution_stats[column] = stats
                
                print(f"     {column}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
                      f"range={stats['range']:.1f}, skew={stats['skewness']:.2f}")
        
        # Apply different scaling methods
        scaled_data = processed_data.copy()
        
        # 1. Standard Scaling (Z-score normalization)
        print(f"\n   📐 Applying standard scaling:")
        standard_scale_columns = ['age', 'years_customer', 'support_tickets']
        
        for column in standard_scale_columns:
            if column in scaled_data.columns:
                original_mean = scaled_data[column].mean()
                original_std = scaled_data[column].std()
                
                # Standard scaling: (x - mean) / std
                scaled_data[f'{column}_standard'] = (
                    (scaled_data[column] - original_mean) / original_std
                )
                
                new_mean = scaled_data[f'{column}_standard'].mean()
                new_std = scaled_data[f'{column}_standard'].std()
                
                print(f"     {column}: mean {original_mean:.1f}→{new_mean:.3f}, "
                      f"std {original_std:.1f}→{new_std:.3f}")
        
        # 2. Min-Max Scaling (0-1 normalization)
        print(f"\n   📏 Applying min-max scaling:")
        minmax_scale_columns = ['credit_score', 'transaction_count']
        
        for column in minmax_scale_columns:
            if column in scaled_data.columns:
                col_min = scaled_data[column].min()
                col_max = scaled_data[column].max()
                col_range = col_max - col_min
                
                # Min-max scaling: (x - min) / (max - min)
                scaled_data[f'{column}_minmax'] = (
                    (scaled_data[column] - col_min) / col_range if col_range > 0 else 0
                )
                
                new_min = scaled_data[f'{column}_minmax'].min()
                new_max = scaled_data[f'{column}_minmax'].max()
                
                print(f"     {column}: range [{col_min:.1f}, {col_max:.1f}] → [{new_min:.3f}, {new_max:.3f}]")
        
        # 3. Robust Scaling (using median and IQR)
        print(f"\n   🛡️ Applying robust scaling:")
        robust_scale_columns = ['income', 'account_balance']
        
        for column in robust_scale_columns:
            if column in scaled_data.columns:
                col_median = scaled_data[column].median()
                q75 = scaled_data[column].quantile(0.75)
                q25 = scaled_data[column].quantile(0.25)
                iqr = q75 - q25
                
                # Robust scaling: (x - median) / IQR
                scaled_data[f'{column}_robust'] = (
                    (scaled_data[column] - col_median) / iqr if iqr > 0 else 0
                )
                
                new_median = scaled_data[f'{column}_robust'].median()
                new_iqr = (scaled_data[f'{column}_robust'].quantile(0.75) - 
                          scaled_data[f'{column}_robust'].quantile(0.25))
                
                print(f"     {column}: median {col_median:.1f}→{new_median:.3f}, "
                      f"IQR {iqr:.1f}→{new_iqr:.3f}")
        
        # 4. Log transformation for skewed data
        print(f"\n   📈 Applying log transformation for skewed data:")
        
        for column in ['income', 'account_balance']:
            if column in scaled_data.columns:
                skewness = distribution_stats.get(column, {}).get('skewness', 0)
                
                if abs(skewness) > 1:  # High skewness
                    # Apply log transformation (add small constant to handle zeros/negatives)
                    min_val = scaled_data[column].min()
                    offset = abs(min_val) + 1 if min_val <= 0 else 0
                    
                    scaled_data[f'{column}_log'] = (
                        (scaled_data[column] + offset).apply(lambda x: math.log(x) if x > 0 else 0)
                    )
                    
                    new_skewness = scaled_data[f'{column}_log'].skew() if hasattr(scaled_data[f'{column}_log'], 'skew') else 0
                    
                    print(f"     {column}: skewness {skewness:.2f}→{new_skewness:.2f} "
                          f"(offset: {offset:.1f})")
                else:
                    print(f"     {column}: skewness {skewness:.2f} (no transformation needed)")
        
        # Summary
        original_columns = len([col for col in scaled_data.columns if col in processed_data.columns])
        scaled_columns = len([col for col in scaled_data.columns if col not in processed_data.columns])
        
        print(f"\n   ✅ Scaling and normalization completed:")
        print(f"     Original numeric columns: {original_columns}")
        print(f"     Scaled versions created: {scaled_columns}")
        print(f"     Total columns: {len(scaled_data.columns)}")
        
        # Show scaling effectiveness
        print(f"\n   📊 Scaling effectiveness:")
        for method in ['standard', 'minmax', 'robust', 'log']:
            method_columns = [col for col in scaled_data.columns if col.endswith(f'_{method}')]
            if method_columns:
                print(f"     {method.capitalize()} scaled: {len(method_columns)} columns")
        
    else:
        print("   🎭 Mock scaling and normalization (pandas not available):")
        print("   Analyzing data distributions:")
        print("     age: mean=45.2, std=15.8, range=62.0, skew=0.12")
        print("     income: mean=78450.0, std=35200.0, range=130000.0, skew=1.85")
        print("     credit_score: mean=675.0, std=125.0, range=550.0, skew=-0.05")
        print("   Applying standard scaling:")
        print("     age: mean 45.2→0.000, std 15.8→1.000")
        print("   Applying min-max scaling:")
        print("     credit_score: range [300.0, 850.0] → [0.000, 1.000]")
        print("   Applying robust scaling:")
        print("     income: median 78450.0→0.000, IQR 35200.0→1.000")
        print("   Applying log transformation:")
        print("     income: skewness 1.85→0.23 (reduced skewness)")


def pipeline_composition_examples():
    """Demonstrate complex pipeline composition and chaining."""
    print("\n🔗 Pipeline Composition")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Pipeline composition examples:
from refunc.data_science import (
    TransformationPipeline, create_basic_pipeline, create_robust_pipeline
)

# Create comprehensive preprocessing pipeline
pipeline = TransformationPipeline(name="ml_preprocessing")

# Step 1: Data validation and cleaning
pipeline.add_step('validate_data', DataValidator())
pipeline.add_step('remove_outliers', OutlierRemover(method='iqr'))

# Step 2: Missing value handling
pipeline.add_step('impute_numeric', MissingValueImputer(strategy='median'))
pipeline.add_step('impute_categorical', MissingValueImputer(strategy='mode'))

# Step 3: Feature engineering
pipeline.add_step('engineer_features', FeatureEngineer())

# Step 4: Encoding and scaling
pipeline.add_step('encode_categorical', CategoricalEncoder(method='onehot'))
pipeline.add_step('scale_numeric', DataScaler(method='robust'))

# Apply complete pipeline
result = pipeline.fit_transform(df)
print(f"Pipeline applied {len(result.steps_applied)} steps")
print(f"Transformation time: {result.execution_time:.2f}s")

# Quick preprocessing alternative
quick_result = apply_quick_preprocessing(df)
        """)
        return
    
    print("🏗️ Testing pipeline composition:")
    
    sample_data = create_sample_dataset()
    
    if PANDAS_AVAILABLE:
        print(f"   📊 Starting pipeline with data: {sample_data.shape}")
        
        # Create a comprehensive preprocessing pipeline
        pipeline_start_time = time.time()
        processed_data = sample_data.copy()
        
        # Pipeline Step 1: Data Quality Assessment
        print(f"\n   1️⃣ Data Quality Assessment:")
        initial_missing = processed_data.isnull().sum().sum()
        initial_shape = processed_data.shape
        print(f"     Initial data: {initial_shape}, missing values: {initial_missing}")
        
        # Pipeline Step 2: Outlier Detection and Handling
        print(f"\n   2️⃣ Outlier Detection and Handling:")
        outlier_columns = ['age', 'income', 'account_balance']
        outlier_stats = {}
        
        for column in outlier_columns:
            if column in processed_data.columns:
                col_data = processed_data[column].dropna()
                if len(col_data) > 0:
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_count = len(outliers)
                    
                    # Cap outliers instead of removing (preserves data)
                    processed_data.loc[processed_data[column] < lower_bound, column] = lower_bound
                    processed_data.loc[processed_data[column] > upper_bound, column] = upper_bound
                    
                    outlier_stats[column] = {
                        'count': outlier_count,
                        'bounds': (lower_bound, upper_bound)
                    }
                    
                    print(f"     {column}: {outlier_count} outliers capped to [{lower_bound:.1f}, {upper_bound:.1f}]")
        
        # Pipeline Step 3: Missing Value Imputation
        print(f"\n   3️⃣ Missing Value Imputation:")
        numeric_columns = processed_data.select_dtypes(include=['number']).columns
        categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
        
        imputation_summary = {}
        
        # Numeric imputation
        for column in numeric_columns:
            if processed_data[column].isnull().any():
                missing_count = processed_data[column].isnull().sum()
                
                # Choose imputation strategy based on data characteristics
                if column in ['age', 'credit_score']:
                    # Use median for age-like variables
                    fill_value = processed_data[column].median()
                    strategy = 'median'
                elif column in ['years_customer', 'transaction_count', 'support_tickets']:
                    # Use mode for count variables
                    fill_value = processed_data[column].mode().iloc[0] if len(processed_data[column].mode()) > 0 else 0
                    strategy = 'mode'
                else:
                    # Use mean for continuous variables
                    fill_value = processed_data[column].mean()
                    strategy = 'mean'
                
                processed_data[column] = processed_data[column].fillna(fill_value)
                imputation_summary[column] = {
                    'strategy': strategy,
                    'fill_value': fill_value,
                    'imputed_count': missing_count
                }
                
                print(f"     {column}: {missing_count} values imputed with {strategy} ({fill_value:.1f})")
        
        # Categorical imputation
        for column in categorical_columns:
            if column in processed_data.columns and processed_data[column].isnull().any():
                missing_count = processed_data[column].isnull().sum()
                
                # Use mode for categorical variables
                mode_value = processed_data[column].mode()
                fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'Unknown'
                
                processed_data[column] = processed_data[column].fillna(fill_value)
                imputation_summary[column] = {
                    'strategy': 'mode',
                    'fill_value': fill_value,
                    'imputed_count': missing_count
                }
                
                print(f"     {column}: {missing_count} values imputed with mode ('{fill_value}')")
        
        # Pipeline Step 4: Feature Engineering (simplified)
        print(f"\n   4️⃣ Feature Engineering:")
        feature_count_before = len(processed_data.columns)
        
        # Create a few key engineered features
        if 'income' in processed_data.columns and 'account_balance' in processed_data.columns:
            processed_data['balance_income_ratio'] = processed_data['account_balance'] / processed_data['income']
            print(f"     Created balance_income_ratio")
        
        if 'transaction_count' in processed_data.columns and 'years_customer' in processed_data.columns:
            processed_data['avg_transactions_per_year'] = processed_data['transaction_count'] / (processed_data['years_customer'] + 1)
            print(f"     Created avg_transactions_per_year")
        
        feature_count_after = len(processed_data.columns)
        new_features = feature_count_after - feature_count_before
        print(f"     Added {new_features} engineered features")
        
        # Pipeline Step 5: Categorical Encoding
        print(f"\n   5️⃣ Categorical Encoding:")
        categorical_columns = ['region', 'product_type']
        
        for column in categorical_columns:
            if column in processed_data.columns:
                # One-hot encoding
                dummies = pd.get_dummies(processed_data[column], prefix=column, drop_first=True)
                processed_data = pd.concat([processed_data, dummies], axis=1)
                processed_data = processed_data.drop(column, axis=1)
                
                print(f"     {column}: one-hot encoded into {len(dummies.columns)} columns")
        
        # Pipeline Step 6: Numeric Scaling
        print(f"\n   6️⃣ Numeric Feature Scaling:")
        numeric_columns_to_scale = processed_data.select_dtypes(include=['number']).columns
        numeric_columns_to_scale = [col for col in numeric_columns_to_scale if not col.startswith(('region_', 'product_type_'))]
        
        scaling_summary = {}
        for column in numeric_columns_to_scale[:5]:  # Scale first 5 numeric columns
            if column in processed_data.columns:
                original_mean = processed_data[column].mean()
                original_std = processed_data[column].std()
                
                # Apply robust scaling (median and IQR)
                median_val = processed_data[column].median()
                q75 = processed_data[column].quantile(0.75)
                q25 = processed_data[column].quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 0:
                    processed_data[f'{column}_scaled'] = (processed_data[column] - median_val) / iqr
                    scaling_summary[column] = 'robust'
                    print(f"     {column}: robust scaled (median={median_val:.1f}, IQR={iqr:.1f})")
        
        # Pipeline Summary
        pipeline_end_time = time.time()
        pipeline_duration = pipeline_end_time - pipeline_start_time
        
        print(f"\n   ✅ Pipeline Execution Summary:")
        print(f"     Steps completed: 6")
        print(f"     Execution time: {pipeline_duration:.2f} seconds")
        print(f"     Initial shape: {initial_shape}")
        print(f"     Final shape: {processed_data.shape}")
        print(f"     Data expansion: {(processed_data.shape[1] / initial_shape[1] - 1) * 100:.1f}%")
        print(f"     Missing values: {initial_missing} → {processed_data.isnull().sum().sum()}")
        
        # Pipeline validation
        print(f"\n   🔍 Pipeline Validation:")
        final_missing = processed_data.isnull().sum().sum()
        numeric_columns_final = len(processed_data.select_dtypes(include=['number']).columns)
        categorical_columns_final = len(processed_data.select_dtypes(include=['object', 'category']).columns)
        
        print(f"     ✓ No missing values: {final_missing == 0}")
        print(f"     ✓ Numeric columns: {numeric_columns_final}")
        print(f"     ✓ Categorical columns: {categorical_columns_final}")
        print(f"     ✓ Ready for ML: {final_missing == 0 and processed_data.shape[1] > initial_shape[1]}")
        
    else:
        print("   🎭 Mock pipeline composition (pandas not available):")
        print("   Starting pipeline with data: (500, 12)")
        print("   1️⃣ Data Quality Assessment: 500 records, 127 missing values")
        print("   2️⃣ Outlier Detection: 15 outliers capped across 3 columns")
        print("   3️⃣ Missing Value Imputation: 127 values imputed using median/mode/mean")
        print("   4️⃣ Feature Engineering: Added 3 engineered features")
        print("   5️⃣ Categorical Encoding: 2 columns one-hot encoded into 7 columns")
        print("   6️⃣ Numeric Scaling: 8 columns robust scaled")
        print("   ✅ Pipeline completed: 6 steps in 0.85 seconds")
        print("   Final shape: (500, 22), Data expansion: 83.3%, No missing values")


def main():
    """Run all preprocessing pipeline examples."""
    print("🚀 Refunc Preprocessing Pipeline Examples")
    print("=" * 65)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    if not PANDAS_AVAILABLE:
        print("⚠️  Pandas not available - using mock data structures")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    basic_transformation_examples()
    feature_engineering_examples()
    scaling_normalization_examples()
    pipeline_composition_examples()
    
    print("\n✅ Preprocessing pipeline examples completed!")
    print("\n📖 Next steps:")
    print("- Implement preprocessing pipelines in your ML workflows")
    print("- Customize transformation steps for your specific data")
    print("- Combine with validation for robust data preparation")
    print("- Check out statistical_analysis.py for advanced data analysis")


if __name__ == "__main__":
    main()