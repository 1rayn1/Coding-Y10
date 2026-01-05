import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sys
import time
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import json

save_file = "players.json"
play = 0

#csv is https://www.kaggle.com/datasets/vijayvvenkitesh/global-land-temperatures-by-country

def write(message):  # Gradually types out every line instead of typewriting it in blocks
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        if char != "\n":
            time.sleep(0.01)
        else:
            time.sleep(0.10)


def save_player_score(player_name, score_delta, filename=save_file):
    # Try to open the save file and load existing player scores
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                # Attempt to parse the file as JSON (dictionary of player scores)
                players = json.load(f)
            except json.JSONDecodeError:
                # If the file exists but is empty or corrupted, start fresh
                players = {}
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty dictionary
        players = {}

    # Get the previous score for the player, defaulting to 0 if not found
    prev = players.get(player_name, 0)
    try:
        # Ensure the previous score is treated as a float
        prev = float(prev)
    except (TypeError, ValueError):
        # If conversion fails (e.g., invalid data), reset to 0.0
        prev = 0.0

    # Add the score change (delta) to the previous score
    new_total = prev + float(score_delta)

    # Store the new score:
    # - If it's a whole number, save as int
    # - Otherwise, keep as float
    players[player_name] = int(new_total) if float(new_total).is_integer() else float(new_total)

    # Write the updated scores back to the file in JSON format
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(players, f, indent=2, ensure_ascii=False)

    # Print confirmation of the saved score
    print(f"Saved: {player_name} -> {players[player_name]}")

    # Return the player's updated score
    return players[player_name]


def year(iowa_model):
    future_dates_1 = pd.date_range(start='2030-01-01', end='2030-12-01', freq='MS')
    future_ordinals_1 = future_dates_1.map(lambda x: x.toordinal())

    future_X_1 = pd.DataFrame({
        'dt_ordinal': future_ordinals_1,
        'month': None
    })

    # Predictions
    future_predictions_1 = iowa_model.predict(future_X_1)
    results = []
    for date, temp in zip(future_dates_1.strftime('%Y-%m-%d'), future_predictions_1):
        results.append(f"{date}: {temp:.2f}")

    return (results[0])[12:]


def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def missing_value_handler(X_train, X_valid, y_train, y_valid, score_dataset):
    my_imputer = SimpleImputer()

    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    X_valid_plus = X_valid.copy()
    X_train_plus = X_train.copy()

    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    method_1 = score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)

    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    method_2 = score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)

    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    method_3 = score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)

    return min(method_1, method_2, method_3)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

player_name = input("Please input your name(for game purposes): ")


# Chooses the file path
csv_file_path = 'GlobalWarming.csv'

# Reads the file into a variable
data_all = pd.read_csv(csv_file_path)

# finds all the unique cities
wanted_city = data_all['City'].unique()
wanted_city = list(wanted_city)

# writes in cool format:
write("Please choose your city below:\n")
write("Note: If you want to exit the program, enter control + c\n")

while True:
    a = input("Enter city: ")
    if a not in wanted_city:
        write("Not a valid name. Try again\n")
        write("Try capitalizing the name.\n")
    else:
        city = a
        break

# set a variable as the wanted city (make a copy so we don't modify original)
data_1 = data_all[data_all['City'] == city].copy()

# Convert 'dt' to datetime first
data_1['dt'] = pd.to_datetime(data_1['dt'], errors='coerce')

# Diagnostics
print("Rows for city:", city)
print("Missing dt:", int(data_1['dt'].isna().sum()))
print("Missing AverageTemperature:", int(data_1['AverageTemperature'].isna().sum()))

# Drop rows missing either dt or AverageTemperature
data_1 = data_1.dropna(subset=['dt', 'AverageTemperature']).copy()

# Create features
data_1['dt_ordinal'] = data_1['dt'].map(lambda x: x.toordinal())
data_1['month'] = data_1['dt'].dt.month  # NEW FEATURE

X = data_1[['dt_ordinal', 'month']]
y = data_1['AverageTemperature']

# Example: January rows
january_data = data_1[data_1['dt'].dt.month == 1]
print("January rows:", len(january_data))
print("To advance to the next step, close the graph")
# Sort by date
data_1 = data_1.sort_values(by='dt')

# extract the year
data_1['year'] = data_1['dt'].dt.year
data_1['year_group'] = (data_1['year'] // 50) * 50

# Group by 50-year averages
grouped = data_1.groupby('year_group')['AverageTemperature'].mean().reset_index()

# Line of best fit
x_data = pd.to_datetime(grouped['year_group'], format='%Y').map(lambda x: x.toordinal())
y_data = grouped['AverageTemperature']

slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
line_of_best_fit = slope * x_data + intercept

plt.figure(figsize=(10, 6))
plt.scatter(grouped['year_group'], y_data, label='50-Year Average Data', color='blue')
plt.plot(grouped['year_group'], line_of_best_fit, color='red', label='Line of Best Fit')

plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.title(f'Average Temperature Over Time (50-Year Averages) of {city}')
plt.legend()
plt.grid(True)
plt.show()


# Machine Learning Part
try:
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    train_X_ = train_X.copy()
    val_X_ = val_X.copy()
    train_y_ = train_y.copy()
    val_y_ = val_y.copy()

    iowa_model_1 = DecisionTreeRegressor(random_state=1)
    iowa_model_1.fit(train_X_, train_y_)
    val_predictions_1 = iowa_model_1.predict(val_X_)
    val_mae_1 = mean_absolute_error(val_y_, val_predictions_1)

    iowa_model_ = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model_.fit(train_X, train_y)
    val_predictions_ = iowa_model_.predict(val_X)
    val_mae_ = mean_absolute_error(val_y, val_predictions_)

    if val_mae_1 >= val_mae_:
        iowa_model = iowa_model_1
    else:
        iowa_model = iowa_model_

    game = float(input("Guess the average temperature in the year 2030: "))
    score_formula = round(30 - abs(float(year(iowa_model)) - game))
    if score_formula < 0:
        score_formula = 0
    play += score_formula

    # Persist the player's score (adds this round's points to previous total)
    total_for_player = save_player_score(player_name, score_formula, save_file)
    print(f"Your round score: {score_formula} | Total saved score for {player_name}: {total_for_player}")

    print("\nPredicted Average Temperature for 2030:")
    print(year(iowa_model))

    # Predict future months in 2030
    future_dates = pd.date_range(start='2030-01-01', end='2030-12-01', freq='MS')
    future_ordinals = future_dates.map(lambda x: x.toordinal())
    future_months = future_dates.month

    future_X = pd.DataFrame({
        'dt_ordinal': future_ordinals,
        'month': future_months
    })

    future_predictions = iowa_model.predict(future_X)
    print("\nPredicted Average Temperatures for 2030 (month-aware):")
    for date, temp in zip(future_dates.strftime('%Y-%m-%d'), future_predictions):
        print(f"{date}: {temp:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_predictions, marker='o', color='green', label='2030 Predictions')
    plt.xlabel('Month')
    plt.ylabel('Predicted Average Temperature (°C)')
    plt.title(f'Predicted Monthly Temperatures for 2030 in {city}')
    plt.legend()
    plt.grid(True)

except FileNotFoundError:
    print("Error: The file was not found. Check the path and filename.")
except ValueError as ve:
    print("ValueError:", ve)
except Exception as e:
    print("An unexpected error occurred:", e)

print("Currect Standings: ")
with open("players.json", "r") as json_file:
    data = json.load(json_file)


print(json.dumps(data, indent=2))

plt.show()