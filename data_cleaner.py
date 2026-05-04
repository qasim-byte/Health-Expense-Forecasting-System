import pandas as pd
from abc import ABC, abstractmethod

# Abstract base class for data processing
class DataProcessor(ABC):
    def __init__(self, data=None):
        self._data = data  # Encapsulated data

    @abstractmethod
    def process(self):
        pass

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

# Concrete class for loading data
class DataLoader(DataProcessor):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path

    def process(self):
        if self.file_path:
            self._data = pd.read_csv(self.file_path)
        return self._data

# Concrete class for cleaning data
class DataCleaner(DataProcessor):
    def __init__(self, data=None):
        super().__init__(data)

    def process(self):
        df = self._data.copy()
        if "insurance_type" in df.columns:
            default_insurance = df["insurance_type"].mode()[0] if not df["insurance_type"].mode().empty else "None"
            df["insurance_type"] = df["insurance_type"].fillna(default_insurance)

        numeric_columns = ["insurance_coverage_pct", "sleep_hours", "daily_steps", "bmi", "previous_year_cost"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if col != "previous_year_cost" else 0)

        self._data = df
        return self._data

# Concrete class for preprocessing data (inherits from DataCleaner)
class DataPreprocessor(DataCleaner):
    CATEGORICAL_MAPPINGS = {
        "gender": {"Male": 0, "Female": 1},
        "smoker": {"No": 0, "Yes": 1},
        "diabetes": {"No": 0, "Yes": 1},
        "hypertension": {"No": 0, "Yes": 1},
        "heart_disease": {"No": 0, "Yes": 1},
        "asthma": {"No": 0, "Yes": 1},
        "physical_activity_level": {"Low": 0, "Medium": 1, "High": 2},
        "city_type": {"Rural": 0, "Semi-Urban": 1, "Urban": 2},
        "insurance_type": {"None": 0, "Government": 1, "Private": 2},
    }

    def process(self):
        super().process()  # Call parent's process (cleaning)
        df = self._data
        for column, mapping in self.CATEGORICAL_MAPPINGS.items():
            if column in df.columns:
                df[column] = df[column].apply(lambda value: self._encode_value(value, mapping))
        self._data = df
        return self._data

    def _encode_value(self, value, mapping):
        if pd.isna(value):
            return -1
        if value in mapping:
            return mapping[value]
        text = str(value).strip()
        if text in mapping:
            return mapping[text]
        try:
            return int(float(text))
        except (ValueError, TypeError):
            return -1

    def set_current_column(self, column):
        self._current_column = column

# Factory function for polymorphism
def create_data_processor(processor_type, data=None, file_path=None):
    if processor_type == "loader":
        return DataLoader(file_path)
    elif processor_type == "cleaner":
        return DataCleaner(data)
    elif processor_type == "preprocessor":
        return DataPreprocessor(data)
    else:
        raise ValueError("Unknown processor type")
