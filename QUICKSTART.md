# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Data

```bash
python generate_sample_data.py
```

This creates a sample dataset with 1000 patients and ~15% sepsis rate.

### Step 3: Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Create time-series features
- Handle imbalanced data
- Train the model
- Save results

### Step 4: Evaluate

```bash
python evaluate.py
```

### Step 5: Make Predictions

```bash
python inference.py --data data/raw/sepsis_data.csv --output results/predictions.csv
```

## 📊 Expected Output

After training, you should see:
- Model saved to `models/sepsis_model.pkl`
- Feature importance plot in `results/feature_importance.png`
- Test metrics in `results/test_metrics.csv`

## 🔧 Customize

Edit `configs/config.yaml` to:
- Change model algorithm (xgboost, lightgbm, catboost)
- Adjust hyperparameters
- Modify feature engineering settings
- Change imbalanced learning method

## 📚 Learn More

See `README.md` for detailed documentation.

## 🆘 Troubleshooting

**Issue**: ModuleNotFoundError
- **Solution**: Make sure you're in the project root directory and dependencies are installed

**Issue**: FileNotFoundError for data
- **Solution**: Run `generate_sample_data.py` first

**Issue**: Memory errors
- **Solution**: Reduce `n_patients` in `generate_sample_data.py` or reduce `max_features` in config

