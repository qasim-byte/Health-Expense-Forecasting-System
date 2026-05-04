import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from abc import ABC, abstractmethod

# Abstract base class for models
class BaseModel(ABC):
    def __init__(self, data=None):
        self._data = data
        self._model = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def get_model(self):
        return self._model

    def set_data(self, data):
        self._data = data

# Concrete class for Random Forest model
class RandomForestModel(BaseModel):
    def __init__(self, data=None, n_estimators=100):
        super().__init__(data)
        self.n_estimators = n_estimators

    def train(self):
        if self._data is None or "annual_medical_cost" not in self._data.columns:
            raise ValueError("Data must be provided and include 'annual_medical_cost' column.")

        X = self._data.drop(columns=["annual_medical_cost"])
        y = self._data["annual_medical_cost"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return rmse, r2

    def predict(self, input_data):
        if self._model is None:
            raise ValueError("Model must be trained before prediction.")
        return self._model.predict(input_data)

# Model factory for polymorphism
def create_model(model_type, data=None):
    if model_type == "random_forest":
        return RandomForestModel(data)
    else:
        raise ValueError("Unknown model type")

# Utility functions
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def save_model(model, file_path: str):
    joblib.dump(model.get_model(), file_path)
