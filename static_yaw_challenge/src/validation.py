"""
Validation Module for Static Yaw Challenge

Handles train/validation splits, cross-validation strategies, and evaluation metrics
that account for distribution shift and segment-level predictions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


def create_stratified_split(df: pd.DataFrame,
                           target_col: str = 'yaw_offset',
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split preserving target distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Data with target column
    target_col : str
        Name of target column
    test_size : float
        Proportion for validation set
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    print(f"Training set: {len(train_df):,} samples")
    print(f"Validation set: {len(val_df):,} samples")
    print("\nTarget distribution in training:")
    print(train_df[target_col].value_counts(normalize=True).sort_index())
    print("\nTarget distribution in validation:")
    print(val_df[target_col].value_counts(normalize=True).sort_index())

    return train_df, val_df


def create_distribution_matched_split(df: pd.DataFrame,
                                     target_col: str = 'yaw_offset',
                                     test_distribution: Dict[str, float] = None,
                                     val_size: int = 10000,
                                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation split that matches test data distribution.

    This upsamples production mode data and downsamples low-wind data to match
    the operating condition distribution seen in the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    target_col : str
        Target column name
    test_distribution : dict
        Desired distribution (e.g., {'production_pct': 0.66, 'standby_pct': 0.26})
    val_size : int
        Desired validation set size
    random_state : int
        Random seed

    Returns
    -------
    tuple
        (train_df, val_df) with val_df matching test distribution
    """
    np.random.seed(random_state)

    if test_distribution is None:
        # Default distribution matching test set
        test_distribution = {
            'production_pct': 0.66,  # Status 10
            'standby_pct': 0.26,      # Status 9
            'other_pct': 0.08         # Other statuses
        }

    # Calculate samples needed for each status
    n_production = int(val_size * test_distribution['production_pct'])
    n_standby = int(val_size * test_distribution['standby_pct'])
    n_other = val_size - n_production - n_standby

    # Sample from each group
    production_data = df[df['turbine_status'] == 10]
    standby_data = df[df['turbine_status'] == 9]
    other_data = df[~df['turbine_status'].isin([9, 10])]

    # Sample with replacement if needed
    val_production = production_data.sample(
        n=min(n_production, len(production_data)),
        replace=n_production > len(production_data),
        random_state=random_state
    )

    val_standby = standby_data.sample(
        n=min(n_standby, len(standby_data)),
        replace=n_standby > len(standby_data),
        random_state=random_state + 1
    )

    val_other = other_data.sample(
        n=min(n_other, len(other_data)),
        replace=n_other > len(other_data),
        random_state=random_state + 2
    ) if len(other_data) > 0 else pd.DataFrame()

    # Combine validation set
    val_df = pd.concat([val_production, val_standby, val_other], ignore_index=True)

    # Remaining data for training
    train_df = df[~df.index.isin(val_df.index)].copy()

    print(f"Training set: {len(train_df):,} samples")
    print(f"Validation set: {len(val_df):,} samples")
    print(f"\nStatus distribution in validation:")
    print(val_df['turbine_status'].value_counts(normalize=True).sort_values(ascending=False))
    print(f"\nTarget distribution in validation:")
    print(val_df[target_col].value_counts(normalize=True).sort_index())

    return train_df, val_df


def create_stratified_kfold(n_splits: int = 5,
                           shuffle: bool = True,
                           random_state: int = 42) -> StratifiedKFold:
    """
    Create stratified K-fold cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of folds
    shuffle : bool
        Whether to shuffle data before splitting
    random_state : int
        Random seed

    Returns
    -------
    StratifiedKFold
        Cross-validator object
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )


def evaluate_classification(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: List[str] = None,
                           print_results: bool = True) -> Dict[str, float]:
    """
    Comprehensive classification evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list
        Names of classes for reporting
    print_results : bool
        Whether to print detailed results

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    if class_names is None:
        class_names = [str(int(c)) + '°' for c in sorted(np.unique(y_true))]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None)
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm
    }

    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f'f1_{class_name}'] = f1_per_class[i]
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]

    if print_results:
        print("=" * 70)
        print("CLASSIFICATION EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {accuracy:.4f}")
        print(f"  F1 Score (Macro):   {f1_macro:.4f}")
        print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"  Precision (Macro):  {precision_macro:.4f}")
        print(f"  Recall (Macro):     {recall_macro:.4f}")

        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}:")
            print(f"    F1:        {f1_per_class[i]:.4f}")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")

        print(f"\nConfusion Matrix:")
        print("Predicted →")
        print(f"{'Actual ↓':<10}", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"{name:<10}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>10}", end="")
            print()

        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        print("=" * 70)

    return metrics


def evaluate_segment_predictions(segment_predictions: pd.DataFrame,
                                segment_actuals: pd.DataFrame,
                                segment_id_col: str = 'segment_id',
                                pred_col: str = 'predicted_yaw',
                                actual_col: str = 'yaw_offset',
                                print_results: bool = True) -> Dict[str, float]:
    """
    Evaluate segment-level predictions.

    Parameters
    ----------
    segment_predictions : pd.DataFrame
        DataFrame with segment IDs and predictions
    segment_actuals : pd.DataFrame
        DataFrame with segment IDs and actual values
    segment_id_col : str
        Column name for segment ID
    pred_col : str
        Column name for predictions
    actual_col : str
        Column name for actual values
    print_results : bool
        Whether to print results

    Returns
    -------
    dict
        Evaluation metrics
    """
    # Merge predictions with actuals
    merged = segment_predictions.merge(
        segment_actuals[[segment_id_col, actual_col]],
        on=segment_id_col,
        how='inner'
    )

    y_true = merged[actual_col].values
    y_pred = merged[pred_col].values

    return evaluate_classification(y_true, y_pred, print_results=print_results)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.

    Uses sklearn's balanced weight formula:
    weight = n_samples / (n_classes * n_samples_per_class)

    Parameters
    ----------
    y : np.ndarray
        Target labels

    Returns
    -------
    dict
        Dictionary mapping class labels to weights
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    class_weights = {}
    for cls, count in zip(unique_classes, class_counts):
        class_weights[cls] = n_samples / (n_classes * count)

    print("Class Weights (for handling imbalance):")
    for cls, weight in class_weights.items():
        print(f"  Class {int(cls)}°: {weight:.4f}")

    return class_weights


def cross_validate_model(model,
                        X: pd.DataFrame,
                        y: np.ndarray,
                        cv: StratifiedKFold,
                        metric_name: str = 'f1_macro') -> Dict[str, List[float]]:
    """
    Perform cross-validation and return metrics for each fold.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate (must have fit() and predict() methods)
    X : pd.DataFrame
        Features
    y : np.ndarray
        Target labels
    cv : StratifiedKFold
        Cross-validator
    metric_name : str
        Metric to optimize (accuracy, f1_macro, f1_weighted)

    Returns
    -------
    dict
        Dictionary with metrics for each fold
    """
    fold_metrics = {
        'accuracy': [],
        'f1_macro': [],
        'f1_weighted': []
    }

    print(f"\nPerforming {cv.n_splits}-fold cross-validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)

        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        f1_mac = f1_score(y_val, y_pred, average='macro')
        f1_wei = f1_score(y_val, y_pred, average='weighted')

        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1_macro'].append(f1_mac)
        fold_metrics['f1_weighted'].append(f1_wei)

        print(f"  Fold {fold_idx + 1}: Accuracy={acc:.4f}, "
              f"F1(macro)={f1_mac:.4f}, F1(weighted)={f1_wei:.4f}")

    # Print summary
    print("\nCross-Validation Summary:")
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    return fold_metrics


def filter_by_operating_condition(df: pd.DataFrame,
                                 min_wind_speed: float = 3.0,
                                 min_power: float = 0.5) -> pd.DataFrame:
    """
    Filter data to focus on operating conditions where yaw effect is strongest.

    Parameters
    ----------
    df : pd.DataFrame
        Data with wind_speed and power_output
    min_wind_speed : float
        Minimum wind speed threshold (m/s)
    min_power : float
        Minimum power output threshold (kW)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    original_len = len(df)

    filtered = df[
        (df['wind_speed'] >= min_wind_speed) &
        (df['power_output'] >= min_power)
    ].copy()

    print(f"Filtered from {original_len:,} to {len(filtered):,} rows "
          f"({len(filtered)/original_len*100:.1f}% retained)")
    print(f"Conditions: wind_speed >= {min_wind_speed} m/s, "
          f"power_output >= {min_power} kW")

    return filtered


def create_test_submission_template(test_df: pd.DataFrame,
                                   predictions: np.ndarray,
                                   row_id_col: str = 'row_id',
                                   output_file: str = 'submission.csv') -> pd.DataFrame:
    """
    Create submission file for the challenge.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with row IDs
    predictions : np.ndarray
        Predicted yaw offsets
    row_id_col : str
        Column name for row IDs
    output_file : str
        Output file path

    Returns
    -------
    pd.DataFrame
        Submission dataframe
    """
    submission = pd.DataFrame({
        row_id_col: test_df[row_id_col],
        'yaw_offset': predictions.astype(int)
    })

    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to: {output_file}")
    print(f"Predictions shape: {submission.shape}")
    print(f"\nPredicted yaw offset distribution:")
    print(submission['yaw_offset'].value_counts().sort_index())

    return submission


# Example usage documentation
if __name__ == "__main__":
    print("Validation Module for Static Yaw Challenge")
    print("=" * 60)
    print("\nKey Functions:")
    print("  - create_stratified_split(): Standard train/val split")
    print("  - create_distribution_matched_split(): Match test distribution")
    print("  - create_stratified_kfold(): K-fold cross-validation")
    print("  - evaluate_classification(): Comprehensive metrics")
    print("  - evaluate_segment_predictions(): Segment-level evaluation")
    print("  - calculate_class_weights(): Handle class imbalance")
    print("  - cross_validate_model(): Cross-validation with metrics")
    print("  - create_test_submission_template(): Generate submission file")
