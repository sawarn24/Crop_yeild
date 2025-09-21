import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import pickle
import os

def train_xgboost_model():
    """Train XGBoost model for crop yield prediction using your exact features"""
    
    print("Loading dataset...")
    try:
        df = pd.read_csv('Custom_Crops_yield_Historical_Dataset.csv')
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("Error: Custom_Crops_yield_Historical_Dataset.csv not found!")
        return None
    
    print(f"Dataset columns: {list(df.columns)}")
    
    # Handle missing values
    df = df.dropna()
    print(f"After removing missing values: {df.shape[0]} rows")
    
    # Define features (X) and target (y) - EXACTLY like your original model
    feature_columns = [
        'Year', 'State Name', 'Dist Name', 'Crop', 'Area_ha',
        'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm', 
        'Wind_Speed_m_s'
    ]
    
    target_column = 'Yield_kg_per_ha'
    
    print(f"\n🎯 TARGET: {target_column}")
    print(f"📊 FEATURES: {feature_columns}")
    
    # Select available columns from your dataset
    available_features = [col for col in feature_columns if col in df.columns]
    
    print(f"\nAvailable features in dataset: {available_features}")
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
    
    # Check if target column exists
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    X = df[available_features]
    y = df[target_column]
    
    print(f"\nFeature matrix X: {X.shape}")
    print(f"Target vector y: {y.shape}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f} kg/ha")
    
    # Handle categorical variables - EXACTLY like your original
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_columns}")
    print(f"Numerical columns: {numerical_columns}")
    
    # Label encode categorical variables - EXACTLY like your original
    label_encoders = {}
    for col in categorical_columns:
        print(f"Encoding {col}...")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")
    
    # Scale numerical features - EXACTLY like your original
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    print("✓ Numerical features scaled")
    
    # Split the data - EXACTLY like your original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ===== REPLACE RANDOM FOREST WITH XGBOOST =====
    print("\nTraining XGBoost model (instead of Random Forest)...")
    
    # XGBoost with similar parameters to your Random Forest
    model = xgb.XGBRegressor(
        n_estimators=100,       # Same as your RF n_estimators
        max_depth=10,          # Reduced from your RF max_depth=20 for efficiency
        learning_rate=0.1,     # XGBoost learning rate
        subsample=0.8,         # Similar to RF bootstrap
        colsample_bytree=0.8,  # Similar to RF max_features
        min_child_weight=2,    # Similar to your min_samples_leaf=2
        gamma=0.1,             # Regularization
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=0.1,        # L2 regularization
        random_state=42,
        n_jobs=-1,             # Use all CPU cores like your RF
        verbosity=0            # Suppress warnings
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\n" + "="*60)
    print("XGBoost Model Performance (vs your Random Forest):")
    print("="*60)
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.2f} kg/ha")
    print(f"Test MAE: {test_mae:.2f} kg/ha")
    print(f"Training RMSE: {train_rmse:.2f} kg/ha")
    print(f"Test RMSE: {test_rmse:.2f} kg/ha")
    
    # Check for overfitting
    if abs(train_r2 - test_r2) > 0.1:
        print("⚠️  Warning: Possible overfitting detected!")
    else:
        print("✅ Model looks good - no significant overfitting")
    
    # Save models and preprocessors - EXACTLY like your original structure
    print("\nSaving models and preprocessors...")
    
    # Save XGBoost model (replaces Random Forest)
    with open('crop_yield_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Saved crop_yield_model.pkl")
    
    # Save preprocessors - EXACTLY like your original
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✓ Saved label_encoders.pkl")
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved scaler.pkl")
    
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(available_features, f)
    print("✓ Saved feature_columns.pkl")
    
    with open('categorical_columns.pkl', 'wb') as f:
        pickle.dump(categorical_columns, f)
    print("✓ Saved categorical_columns.pkl")
    
    with open('numerical_columns.pkl', 'wb') as f:
        pickle.dump(numerical_columns, f)
    print("✓ Saved numerical_columns.pkl")
    
    # Check model sizes
    print("\nModel file sizes:")
    print("-" * 40)
    total_size = 0
    files_to_check = [
        'crop_yield_model.pkl',
        'label_encoders.pkl', 
        'scaler.pkl',
        'feature_columns.pkl',
        'categorical_columns.pkl',
        'numerical_columns.pkl'
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"{filename}: {size_mb:.2f} MB")
            total_size += size_mb
    
    print("-" * 40)
    print(f"Total model size: {total_size:.2f} MB")
    
    if total_size < 30:
        print("✅ Perfect! Model size is under 30MB - great for deployment!")
    elif total_size < 100:
        print("✅ Good! Model size is reasonable for deployment")
    else:
        print("⚠️  Model size is large - but still much smaller than Random Forest")
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop Feature Importances:")
    print("-" * 40)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:<20}: {row['importance']:.4f}")
    
    print(f"\n🎉 XGBoost model training completed successfully!")
    print(f"📊 Final Test R² Score: {test_r2:.4f}")
    print(f"📦 Total Model Size: {total_size:.2f} MB (vs {1160:.0f}MB Random Forest)")
    print(f"🚀 Ready for deployment!")
    
    return model

# Your EXACT prediction function - but using XGBoost model
def predict_yield_from_farmer_input(state_name, district_name, crop, area_ha, weather_api_data, soil_api_data):
    """
    Make yield prediction using farmer input + real-time API data
    Uses XGBoost model instead of Random Forest (same interface)
    
    Parameters:
    - state_name: str (from farmer)
    - district_name: str (from farmer) 
    - crop: str (from farmer)
    - area_ha: float (from farmer)
    - weather_api_data: dict with keys ['temperature', 'humidity', 'rainfall', 'wind_speed']
    - soil_api_data: dict with keys ['ph']
    
    Returns predicted yield in kg/ha
    """
    # Load saved model and preprocessors - EXACTLY like your original
    with open('crop_yield_model.pkl', 'rb') as f:
        model = pickle.load(f)  # Now loads XGBoost instead of Random Forest
    
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('categorical_columns.pkl', 'rb') as f:
        cat_cols = pickle.load(f)
    
    with open('numerical_columns.pkl', 'rb') as f:
        num_cols = pickle.load(f)
    
    # Prepare input data dictionary - EXACTLY like your original
    new_data_dict = {
        'State Name': state_name,
        'Dist Name': district_name,
        'Year': 2025,  # Current year
        'Crop': crop,
        'pH': soil_api_data['ph'],
        'Temperature_C': weather_api_data['temperature'],
        'Humidity_%': weather_api_data['humidity'],
        'Area_ha': area_ha,
        'Rainfall_mm': weather_api_data['rainfall'],
        'Wind_Speed_m_s': weather_api_data['wind_speed']
    }
    
    # Create dataframe from input - EXACTLY like your original
    new_df = pd.DataFrame([new_data_dict])
    
    # Apply same preprocessing - EXACTLY like your original (FIXED)
    for col in cat_cols:
        if col in new_df.columns:
            # Handle unseen categories
            val = str(new_df[col].iloc[0])
            if val not in encoders[col].classes_:
                # Use most common class or first class for unseen values
                print(f"Warning: '{val}' not seen in training for {col}. Using '{encoders[col].classes_[0]}'")
                new_df[col] = encoders[col].classes_[0]
                new_df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
            else:
                new_df[col] = encoders[col].transform([val])[0]
    
    # Scale numerical features - EXACTLY like your original
    new_df[num_cols] = scaler.transform(new_df[num_cols])
    
    # Make prediction - XGBoost instead of Random Forest
    prediction = model.predict(new_df[features])
    return prediction[0]

if __name__ == "__main__":
    # Train the model
    model = train_xgboost_model()
    
    if model is not None:
        print("\n" + "="*60)
        print("Testing prediction with your EXACT example:")
        print("="*60)
        
        # Your EXACT test case
        farmer_input = {
            'state': 'Bihar',
            'district': 'kolkata', 
            'crop': 'maize',
            'area': 2.5
        }
        
        # Weather data (normally from weather API)
        weather_data = {
            'temperature': 25.0,
            'humidity': 80.0,
            'rainfall': 1200.0,
            'wind_speed': 2.0
        }
        
        # Soil data (normally from soil API)
        soil_data = {
            'ph': 6.5
        }
        
        # Make prediction
        try:
            predicted_yield = predict_yield_from_farmer_input(
                farmer_input['state'], 
                farmer_input['district'],
                farmer_input['crop'],
                farmer_input['area'],
                weather_data,
                soil_data
            )
            
            print(f"Farmer Details:")
            print(f"  State: {farmer_input['state']}")
            print(f"  District: {farmer_input['district']}")
            print(f"  Crop: {farmer_input['crop']}")
            print(f"  Area: {farmer_input['area']} hectares")
            print(f"\nWeather Conditions:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Rainfall: {weather_data['rainfall']}mm")
            print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
            print(f"\nSoil Conditions:")
            print(f"  pH: {soil_data['ph']}")
            print(f"\n{'='*50}")
            print(f"XGBoost PREDICTED YIELD: {predicted_yield:.2f} kg/ha")
            print(f"TOTAL EXPECTED PRODUCTION: {predicted_yield * farmer_input['area']:.2f} kg")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            print("Make sure you have trained the model first!")
    
    print("\n✅ XGBoost model ready! Same features, much smaller size!")