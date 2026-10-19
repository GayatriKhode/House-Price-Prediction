"""
House Price Prediction using Linear Regression
Run: python house_price_prediction.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "house_data.csv"
OUTPUT_PLOT = PROJECT_DIR / "actual_vs_predicted.png"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Keep house_data.csv beside this script."
        )

    data = pd.read_csv(DATA_FILE)
    print("\n--- Dataset preview ---")
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    print("\nMissing values by column:")
    print(data.isnull().sum())

    target = "price"
    features = [
        "area_sqft", "bedrooms", "bathrooms",
        "house_age_years", "location_score", "parking_spaces"
    ]
    missing_columns = [col for col in features + [target] if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[features + [target]].dropna()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Results ---")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R² score: {r2:.3f}")

    results = pd.DataFrame({
        "Actual_Price": y_test.to_numpy(),
        "Predicted_Price": predictions.round(0).astype(int)
    })
    results.to_csv(PROJECT_DIR / "predictions.csv", index=False)
    print("\nSaved predictions.csv")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.75)
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual House Price (INR)")
    plt.ylabel("Predicted House Price (INR)")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved graph: {OUTPUT_PLOT.name}")

    # Example prediction using plausible input values
    new_house = pd.DataFrame([{
        "area_sqft": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "house_age_years": 5,
        "location_score": 8,
        "parking_spaces": 1
    }])
    predicted_price = model.predict(new_house)[0]
    print(f"\nExample prediction for a 1,500 sqft, 3-bedroom house: ₹{predicted_price:,.0f}")
    print("\nNote: This is an educational demo dataset, not a real market valuation.")

if __name__ == "__main__":
    main()
