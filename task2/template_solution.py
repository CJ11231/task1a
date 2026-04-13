import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Die neuen, stärkeren Modelle für das Stacking
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

def load_data():
    """
    Loads the training and test data, one-hot encodes seasons, 
    imputes missing features, and drops rows with missing targets.
    """
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_len = len(train_df)
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # drop_first=False ist besser für Baum-Modelle, damit sie jede Saison direkt sehen können
    combined = pd.get_dummies(combined, columns=['season'], drop_first=False)

    train_encoded = combined.iloc[:train_len].copy()
    test_encoded = combined.iloc[train_len:].copy()

    y_train_raw = train_encoded['price_CHF'].values
    train_features = train_encoded.drop(columns=['price_CHF'])
    test_features = test_encoded.drop(columns=['price_CHF'])

    # BayesianRidge ist extrem stabil für die Imputation von stark korrelierten Features (wie Preisen)
    imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=50)
    
    all_features = pd.concat([train_features, test_features], axis=0)
    imputer.fit(all_features)

    X_train_imp = imputer.transform(train_features)
    X_test_imp = imputer.transform(test_features)

    # Zielvariablen (Target) bereinigen
    valid_idx = ~pd.isna(y_train_raw)
    X_train = X_train_imp[valid_idx]
    y_train = y_train_raw[valid_idx]

    X_test = X_test_imp

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        
        # LEVEL 0: Die Basis-Modelle (maximale Diversität)
        
        # 1. State-of-the-Art Tree Booster (schnell und extrem stark)
        hgb = HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.03, l2_regularization=0.1, random_state=42
        )
        
        # 2. Extra Trees (reduziert Varianz, sehr gut bei korrelierten Features durch max_features='sqrt')
        et = ExtraTreesRegressor(
            n_estimators=500, max_depth=15, max_features='sqrt', random_state=42
        )
        
        # 3. Klassischer Random Forest
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=15, max_features='sqrt', random_state=42
        )
        
        # 4. Support Vector Machine (benötigt skalierte Daten)
        svr = Pipeline([
            ('scaler', StandardScaler()), 
            ('svr', SVR(C=10.0, epsilon=0.01))
        ])
        
        # 5. Regularisierte Lineare Regression (als extrem stabiler Anker)
        ridge = Pipeline([
            ('scaler', StandardScaler()), 
            ('ridge', RidgeCV(alphas=np.logspace(-4, 4, 100)))
        ])
        
        # LEVEL 1: Der Meta-Lerner, der lernt, wem er wann vertrauen muss
        final_estimator = RidgeCV(alphas=np.logspace(-4, 4, 100))
        
        # Das ultimative Stacking-Modell (n_jobs=-1 nutzt alle CPU-Kerne deines Macs)
        self.model = StackingRegressor(
            estimators=[
                ('hgb', hgb), 
                ('et', et), 
                ('rf', rf), 
                ('svr', svr), 
                ('ridge', ridge)
            ],
            final_estimator=final_estimator,
            cv=5, # 5-fache Kreuzvalidierung, um Overfitting des Stackers zu verhindern
            n_jobs=-1 
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    
    print(f"Training started on {X_train.shape[0]} samples. This might take 10-30 seconds...")
    model = Model()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("Results file successfully generated! The model is highly optimized now.")