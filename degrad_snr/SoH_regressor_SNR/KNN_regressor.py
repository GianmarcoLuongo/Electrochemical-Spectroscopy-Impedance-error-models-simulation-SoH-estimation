import pandas as pd
import numpy as np
import sys
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error, max_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

np.random.seed(42)

# ========================================
# PARAMETRI DA TERMINALE
# ========================================
DATA_PATH = sys.argv[1]
LABEL = sys.argv[2]

print(f"Dataset: {DATA_PATH}")
print(f"Label: {LABEL}")

# ========================================
# PARTIAL SCALER
# ========================================
class PartialScaler:
    def __init__(self, n_features_to_scale):
        self.scaler = StandardScaler()
        self.n_features_to_scale = n_features_to_scale

    def fit(self, X, y=None):
        self.scaler.fit(X[:, :self.n_features_to_scale])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[:, :self.n_features_to_scale] = self.scaler.transform(X[:, :self.n_features_to_scale])
        return X_copy

    def fit_transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[:, :self.n_features_to_scale] = self.scaler.fit_transform(X[:, :self.n_features_to_scale])
        return X_copy

# ========================================
# CV GENERATOR
# ========================================
def CVGenerator(dataset):
    folds = [
        [28, 29, 2, 12, 17, 15],
        [3, 4, 13, 20, 18, 30],
        [17, 21, 14, 19, 31, 5],
        [15, 23, 24, 32, 6, 22]
    ]
    for batt_test in folds:
        trainingSet = dataset[~dataset['Cell_Name'].isin(batt_test)]
        testSet = dataset[dataset['Cell_Name'].isin(batt_test)]
        yield (trainingSet.index.values.astype(int), testSet.index.values.astype(int))

# ========================================
# CARICA DATASET
# ========================================
df = pd.read_csv(DATA_PATH)
features = [col for col in df.columns if '-Re' in col or '-Im' in col] + ['Temp', 'SoC']
X = df[features].values
X_eis_len = len(features) - 2
y = df['SoH_Actual'].values / 100.0

# ========================================
# MODELLO + GRID SEARCH
# ========================================
pipeline = Pipeline([
    ('scaler', PartialScaler(n_features_to_scale=X_eis_len)),
    ('KNN', KNeighborsRegressor(n_neighbors=10, weights='distance'))
])

param_grid = {
    'KNN__n_neighbors': np.arange(10, 100)
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=CVGenerator(df),
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)
print("Best n_neighbors:", grid.best_estimator_.get_params()['KNN__n_neighbors'])

# ========================================
# CV FINALE
# ========================================
MAEs, R2s, RMSEs, Max_Errors = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(CVGenerator(df)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean

    grid.best_estimator_.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test) + y_train_mean

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)

    MAEs.append(mae)
    R2s.append(r2)
    RMSEs.append(rmse)
    Max_Errors.append(max_err)

    print(f"\nFold {fold+1}:")
    print(f"  MAE = {mae}")
    print(f"  RMSE = {rmse}")
    print(f"  MAX_ERR = {max_err}")
    print(f"  R2 = {r2}")

# ========================================
# RISULTATI E SALVATAGGIO
# ========================================
print("\n" + "="*50)
print("RISULTATI FINALI")
print("="*50)
print(f"Average RMSE: {np.mean(RMSEs):.4f} ± {np.std(RMSEs):.4f}")
print(f"Average MAE:  {np.mean(MAEs):.4f} ± {np.std(MAEs):.4f}")
print(f"Average MAX:  {np.mean(Max_Errors):.4f} ± {np.std(Max_Errors):.4f}")
print(f"Average R2:   {np.mean(R2s):.4f} ± {np.std(R2s):.4f}")

metriche_finali = {
    'RMSE': f"{np.mean(RMSEs):.4f} ± {np.std(RMSEs):.4f}",
    'MAE': f"{np.mean(MAEs):.4f} ± {np.std(MAEs):.4f}",
    'MAX_ERR': f"{np.mean(Max_Errors):.4f} ± {np.std(Max_Errors):.4f}",
    'R2': f"{np.mean(R2s):.4f} ± {np.std(R2s):.4f}",
}

file_name = "metriche_modelli_optimized.csv"
row_name = f"KNN_{LABEL}"

if os.path.exists(file_name):
    df_existing = pd.read_csv(file_name, index_col=0)
    df_existing.loc[row_name] = metriche_finali
    df_existing.to_csv(file_name)
else:
    pd.DataFrame([metriche_finali], index=[row_name]).to_csv(file_name, index=True)

print(f"\nMetriche salvate come '{row_name}' in {file_name}")