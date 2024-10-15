# Data Preprocessing for Consumer Dataset

This project focuses on data preprocessing techniques applied to a consumer dataset. The dataset contains information about regions, age, income, and online shopping behaviors of individuals. The main objective is to clean, transform, and prepare the data for further analysis or machine learning tasks.

## Project Overview

The dataset consists of several features, including:

- **Region**: Country of the individual (India, Brazil, USA)
- **Age**: Age of the individual
- **Income**: Annual income in dollars
- **Online Shopper**: Whether the individual is an online shopper (Yes/No)

We apply several preprocessing steps to handle missing data, encode categorical variables, and standardize numerical features.

## Dataset

The data used for this project is saved in a CSV file. Example of the data:

| Region | Age | Income | Online Shopper |
| ------ | --- | ------ | -------------- |
| India  | 49  | 86400  | No             |
| Brazil | 32  | 57600  | Yes            |
| USA    | 35  | 64800  | No             |
| ...    | ... | ...    | ...            |

## Preprocessing Steps

1. **Handling Missing Data**:
   - Missing values in the `Age` and `Income` columns are filled using the mean of the respective columns.
2. **Encoding Categorical Data**:

   - The `Region` column is label encoded (India = 0, Brazil = 1, USA = 2).
   - The label-encoded `Region` column is then transformed using OneHotEncoding to avoid ordinal relationships.
   - The target variable `Online Shopper` is label encoded (Yes = 1, No = 0).

3. **Feature Scaling**:
   - We apply standardization using `StandardScaler` to ensure all numerical features are on the same scale. This improves the performance of machine learning models that are sensitive to feature scaling.

## Code Example

Here is a brief summary of the main preprocessing steps implemented in the code:

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load dataset
data = pd.read_csv('sample_data.csv')

# Extract features and target
X = data.iloc[: , :-1].values
y = data.iloc[: , -1].values

# Handle missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:] = imputer.fit_transform(X[:, 1:])

# Encode categorical data
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])

# OneHotEncoding for region
column_transformer = ColumnTransformer([('region', OneHotEncoder(), [0])], remainder='passthrough')
X = column_transformer.fit_transform(X)

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

## Requirements

To run the project, you will need to install the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`

You can install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repository-name
   ```

3. Run the code in a Jupyter Notebook or Python environment.
