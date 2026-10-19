# House Price Prediction using Linear Regression

## Project objective
Build a beginner-friendly machine-learning model that predicts house prices from numeric property features.

## Project files
- `house_data.csv` — sample real-estate dataset (300 rows)
- `house_price_prediction.py` — training, evaluation, example prediction, and graph generation
- `requirements.txt` — Python libraries needed
- `README.md` — setup and run instructions

## Requirements
Install Python 3.10 or newer and VS Code (optional).

## Run in VS Code / Terminal
1. Extract the ZIP file.
2. Open the extracted `House_Price_Prediction_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the project:

   ```bash
   python house_price_prediction.py
   ```

   On some Windows systems, use `py` instead of `python`:

   ```bash
   py -m pip install -r requirements.txt
   py house_price_prediction.py
   ```

## Outputs
After running, the project creates:
- `predictions.csv` — actual and predicted prices for the test set
- `actual_vs_predicted.png` — scatter plot comparing actual and predicted prices

## Dataset note
The included CSV is a synthetic sample created for learning and demonstration. Its prices are generated from plausible feature relationships with random noise; they are **not actual property listings or reliable market prices**. For a real-world model, replace it with a properly sourced real-estate dataset and check its column names.

## Method
1. Read CSV using pandas.
2. Select numeric features and target (`price`).
3. Split into 80% training and 20% testing data.
4. Train a `LinearRegression` model.
5. Evaluate using RMSE and R² score.
6. Save predictions and an actual-versus-predicted plot.
