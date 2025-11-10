# Static Yaw Challenge - Analysis Solution

Automated yaw offset detection for Aventa AV-7 wind turbine using SCADA telemetry data.

## Project Overview

**Challenge**: Identify static yaw misalignment (0°, 4°, 6°) from 1 Hz SCADA data without direct yaw measurements.

**Turbine**: Aventa AV-7 (6kW) research turbine at IET-OST, Switzerland
- Rated Power: 6.2 kW
- Rotor Diameter: 12.9 m
- Hub Height: 18.0 m
- Data Period: Jan-Oct 2025 (286 days)

**Key Challenge**: Significant distribution shift between train and test data
- Train: 73% low-wind conditions
- Test: 52% low-wind, 47% operating conditions
- Must generalize to mystery yaw offset in private test

## Project Structure

```
static_yaw_challenge/
├── data/
│   └── yaw_alignment_dataset/
│       ├── train.parquet          # 10M rows, 11 columns, includes yaw_offset
│       ├── test.parquet           # 1M rows, 298 segments
│       └── metadata.croissant.json
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_baseline_models.ipynb  # Baseline modeling
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py    # Feature creation functions
│   └── validation.py             # Validation and evaluation
├── results/                       # Model outputs, predictions
├── reports/
│   └── figures/                  # Visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_eda.ipynb - Data exploration
# 2. notebooks/02_baseline_models.ipynb - Baseline models
```

## Key Findings from EDA

### Target Distribution
- **0° yaw**: 45.9% (4.7M rows) - majority class
- **4° yaw**: 20.9% (2.1M rows) - **underrepresented**
- **6° yaw**: 33.2% (3.4M rows)

### Critical Insights

1. **Relative wind direction shows NO direct signal** (correlation ≈ 0)
   - Yaw offset must be inferred from power loss patterns
   - Requires sophisticated feature engineering

2. **Distribution Shift** (Train → Test):
   - Wind speed: 2.28 → 3.33 m/s (+46%)
   - Power output: 1.05 → 2.03 kW (+93%)
   - Production mode: 54% → 66%
   - Models must generalize across operating conditions

3. **Power Loss Pattern**:
   - Clear power reduction with yaw misalignment
   - 4° and 6° offsets show 5-15% power loss
   - Strongest signal in 5-12 m/s wind range

4. **Weak Feature Correlations**:
   - All SCADA signals show weak correlation with yaw (<0.11)
   - Power output: -0.108 (strongest)
   - Rotor speed: -0.108
   - Wind direction: 0.002 (essentially zero)

## Feature Engineering Approach

### Segment-Level Features
Primary approach: Aggregate to 298 test segments (1-hour each)

**Statistical Aggregations**:
- Mean, std, min, max, median for all SCADA channels
- Percentiles (25th, 50th, 75th)

**Domain-Specific Features**:
- **Power coefficient (Cp)**: Efficiency of power extraction
- **Tip speed ratio (TSR)**: Blade tip speed / wind speed
- **Power loss metrics**: Deviation from baseline power curve
- **Circular statistics**: Wind direction mean, std, dispersion

**Operational Features**:
- % time in each turbine status
- Production mode filtering (status = 10)
- Transition counts

**Temporal Features**:
- Rolling window statistics (10s, 30s, 60s, 300s)
- Rate of change features
- Trend analysis

## Baseline Models

### Approach
- **Level**: Segment-level prediction (298 segments)
- **Features**: ~100+ engineered features per segment
- **Validation**: Stratified split with class weighting

### Models Implemented

1. **Random Forest Classifier**
   - 200 estimators, max_depth=15
   - Class weighting for imbalance
   - Strong baseline performance expected

2. **Logistic Regression**
   - With feature scaling
   - Multi-class classification
   - Class weighting

### Handling Class Imbalance
- Class weights calculated: n_samples / (n_classes × n_samples_per_class)
- Stratified sampling in train/validation split
- Weighted metrics (F1-weighted)

## Evaluation Metrics

**Primary Metrics**:
- **Accuracy**: Overall correctness
- **F1 Score (Macro)**: Average F1 across classes (handles imbalance)
- **F1 Score (Weighted)**: F1 weighted by class support

**Per-Class Metrics**:
- F1, Precision, Recall for each yaw offset (0°, 4°, 6°)
- Confusion matrix analysis

**Segment-Level**:
- 298 segment predictions
- Mapped to row-level for submission

## Key Files

### Notebooks
- **01_eda.ipynb**: Comprehensive exploratory data analysis
  - Target distribution analysis
  - Distribution shift visualization
  - Power curve analysis by yaw offset
  - Feature correlation analysis
  - Segment characteristics

- **02_baseline_models.ipynb**: Baseline modeling
  - Segment-level feature engineering
  - Random Forest and Logistic Regression
  - Model comparison and evaluation
  - Feature importance analysis
  - Test predictions and submission file

### Python Modules

**src/feature_engineering.py**:
- `create_segment_features()`: Primary aggregation function
- `calculate_power_coefficient()`: Cp calculation
- `calculate_tip_speed_ratio()`: TSR calculation
- `calculate_power_loss_features()`: Baseline comparison
- `create_rolling_features()`: Temporal features
- `calculate_circular_statistics()`: Wind direction stats

**src/validation.py**:
- `create_stratified_split()`: Train/val split
- `create_distribution_matched_split()`: Match test distribution
- `evaluate_classification()`: Comprehensive metrics
- `calculate_class_weights()`: Handle imbalance
- `cross_validate_model()`: K-fold CV
- `create_test_submission_template()`: Generate submission

## Next Steps for Improvement

### 1. Advanced Models
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Two-stage approach**: Row-level features → Segment-level classification
- **Ensemble methods**: Blend multiple models
- **Neural networks**: For temporal patterns

### 2. Feature Engineering
- Power curve deviation features
- More sophisticated temporal features
- Interaction features (wind × power, etc.)
- Lag features within segments

### 3. Distribution Shift Handling
- Domain adaptation techniques
- Importance weighting
- Focus on production mode data (status=10)
- Adversarial validation

### 4. Mystery Offset Strategy
- **Regression approach**: Predict continuous yaw angle
- **Outlier detection**: Identify patterns not matching 0°/4°/6°
- **Robust features**: Generalize beyond specific offset values

### 5. Hyperparameter Tuning
- Grid search / Random search
- Bayesian optimization
- Cross-validation strategy

## Data Description

### Training Data (train.parquet)
**Shape**: 10,147,959 rows × 11 columns

**Columns**:
1. `datetime` - Timestamp
2. `rotor_speed` - RPM
3. `generator_speed` - RPM
4. `generator_temperature` - °C
5. `wind_speed` - m/s
6. `power_output` - kW
7. `relative_wind_direction` - degrees
8. `supply_voltage` - V
9. `blade_pitch_deg` - degrees
10. `turbine_status` - Status code (0-13)
11. `yaw_offset` - **TARGET** (0, 4, 6 degrees)

### Test Data (test.parquet)
**Shape**: 1,064,641 rows × 12 columns (298 segments)

**Additional columns**:
- `row_id` - Unique identifier for submission
- `segment_id` - Segment identifier (298 unique)
- `segment_time` - Time within segment (0-3600s)

**Missing columns** (to predict):
- `yaw_offset` - Target variable
- `datetime` - Replaced by segment_time

### Turbine Status Codes
- 0-9: Initialization and standby states
- **10: Power operation (production mode)** ← Focus here
- 11: High wind shutdown
- 12: Normal shutdown
- 13: Alarm/fault condition

## Performance Targets

**Minimum Acceptable**:
- Accuracy > 70%
- F1 (Macro) > 0.65

**Good Performance**:
- Accuracy > 85%
- F1 (Macro) > 0.80

**Excellent Performance**:
- Accuracy > 90%
- F1 (Macro) > 0.85
- Robust to mystery offset

## References

**Dataset Citation**:
```
Barber, S., Hammer, F., & Marykovskiy, Y. (2025).
Aventa AV-7 (6kW) IET-OST Research Wind Turbine SCADA with Static Yaw Offset
[Data set]. Zenodo. https://doi.org/10.5281/zenodo.16276333
```

**Turbine Information**:
- Location: Taggenberg, Switzerland (Lat: 47.52, Long: 8.68236)
- Owner: IET-OST Institute for Energy Technology
- Model: Aventa AV-7
- Type: Upwind, 3-bladed, belt-driven, PMSG

## Contributing

This is an analysis project following the claude.md best practices template.

### Development Workflow
1. Create feature branches for new analysis
2. Use Jupyter notebooks for exploration
3. Move reusable code to src/ modules
4. Document findings in notebooks
5. Update README with key insights

### Code Style
- Follow PEP 8
- Use type hints where appropriate
- Google-style docstrings
- Meaningful variable names
- Comments for complex logic

## License

Dataset: Creative Commons Attribution 3.0 (CC BY 3.0)

## Contact

For questions about the analysis approach, refer to the claude.md template and notebook documentation.

---

**Last Updated**: 2025-10-23
**Version**: 1.0 - Baseline Analysis Solution
