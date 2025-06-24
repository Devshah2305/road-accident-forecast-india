# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(Year, Average_Accident)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

# ✅ Real accident data for India (2016–2021)
data = {
    'Year': [2016, 2017, 2018, 2019, 2020, 2021],
    'Average_Accident': [480652, 464910, 470403, 456959, 372181, 412432]
}
df = pd.DataFrame(data)

# ✅ Log transform to handle big numbers smoothly
df['Log_Accidents'] = np.log(df['Average_Accident'])

# ✅ Polynomial Regression (Degree 3)
X = df[['Year']]
y = df['Log_Accidents']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# ✅ Predict for 2022 to 2025
future_years = pd.DataFrame({'Year': [2022, 2023, 2024, 2025]})
future_poly = poly.transform(future_years)
predicted_log = model.predict(future_poly)
predicted = np.exp(predicted_log)  # Convert back from log scale
predicted = np.maximum(predicted, 0)  # Clip negative predictions

# ✅ Plot actual + predicted
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Average_Accident'], label='Actual (2016–2021)', marker='o', color='blue')
plt.plot(future_years['Year'], predicted, '--', label='Predicted (2022–2025)', marker='x', color='orange')
plt.xlabel('Year')
plt.ylabel(' Average Accident Count')
plt.title('India Road Accident Forecast (2016–2025)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()