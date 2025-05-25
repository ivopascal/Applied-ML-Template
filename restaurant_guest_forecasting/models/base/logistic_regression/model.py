import pickle
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class GuestPredictionLogisticModel:
    """
    A logistic regression model to predict the number of guests.

    This class encapsulates preprocessing, training, prediction, evaluation,
    and model persistence functionalities. It uses a Scikit-learn pipeline
    to handle feature transformations and the logistic regression classifier.

    Raw Features Expected:
    - Date: date (ISO string)
    - is_Friday, is_Monday, ..., is_Wednesday: one-hot encoded (0/1)
    - IsHoliday: one-hot encoded (0/1)
    - Ascension Day, Christmas, ..., Whit Monday: one-hot encoded (0/1)
    - tempmax, tempmin, ..., uvindex: numerical weather data
    - rain, snow: one-hot encoded (0/1) weather conditions
    - art_bloedworst, ..., art_zalmfilet: numerical rank of articles

    Target:
    - Number of guests (treated as discrete classes for logistic regression).
    """

    DEFAULT_MODEL_PATH: Path = Path("guests_logistic_model.pkl")

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        logreg_C: float = 1.0,
        logreg_solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = "lbfgs",
        logreg_max_iter: int = 200,  # Increased for potential convergence
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Initializes the GuestPredictionLogisticModel.

        Args:
            model_path (Optional[Union[str, Path]]): Path to load/save the model.
                                                     Defaults to DEFAULT_MODEL_PATH.
            logreg_C (float): Inverse of regularization strength for Logistic Regression.
                              Smaller values specify stronger regularization.
            logreg_solver (Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]): Algorithm to use in the optimization problem.
                                                                                                          Common options: 'lbfgs', 'liblinear', 'saga'.
            logreg_max_iter (int): Maximum number of iterations for the solver to converge.
            random_state (Optional[int]): Controls the randomness of the model for reproducibility.
                                          Set to an integer for reproducible output.
        """
        self.model_path: Path = (
            Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        )
        self.pipeline_: Optional[Pipeline] = None  # Populated after training or loading
        self.logreg_C: float = logreg_C
        self.logreg_solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = logreg_solver
        self.logreg_max_iter: int = logreg_max_iter
        self.random_state: Optional[int] = random_state

        self._define_feature_sets()

    def _define_feature_sets(self) -> None:
        """Defines the lists of feature names for preprocessing."""
        # Features derived from 'Date'
        self.date_derived_cols_: List[str] = ["year", "month", "day_of_year"]

        # Numerical features from weather data
        self.base_numerical_cols_: List[str] = [
            "tempmax",
            "tempmin",
            "temp",
            "feelslikemax",
            "feelslikemin",
            "feelslike",
            "humidity",
            "precip",
            "precipprob",
            "windspeed",
            "cloudcover",
            "solarradiation",
            "uvindex",
        ]

        # Numerical features from article ranks
        self.art_cols_: List[str] = [
            "art_bloedworst",
            "art_broodplankje",
            "art_captain_dinner",
            "art_carpaccio",
            "art_creme_brulee",
            "art_dame_blanche",
            "art_garnalen_cocktail",
            "art_gehaktbal",
            "art_kaasplankje",
            "art_kalfslever",
            "art_koffie_compleet",
            "art_olijven",
            "art_poffert",
            "art_sate_spies",
            "art_schnitzel",
            "art_sliptong",
            "art_sorbet",
            "art_spareribs",
            "art_stamppot",
            "art_tournedos",
            "art_vers_van_de_markt",
            "art_zalmfilet",
        ]

        # All features that will be numerically processed (imputed and scaled)
        self.all_numerical_cols_for_pipeline_: List[str] = (
            self.date_derived_cols_ + self.base_numerical_cols_ + self.art_cols_
        )

        # Categorical features that are already one-hot encoded
        self.categorical_ohe_cols_: List[str] = [
            "is_Friday",
            "is_Monday",
            "is_Saturday",
            "is_Sunday",
            "is_Thursday",
            "is_Tuesday",
            "is_Wednesday",
            "IsHoliday",
            "Ascension Day",
            "Christmas",
            "Day of German Unity",
            "Easter Monday",
            "Good Friday",
            "King's Day",
            "May Day",
            "New Year's Day",
            "Second Christmas Day",
            "Whit Monday",
            "rain",
            "snow",
        ]

        # All columns expected by the pipeline after the _preprocess step
        self.expected_pipeline_input_cols_: List[str] = (
            self.all_numerical_cols_for_pipeline_ + self.categorical_ohe_cols_
        )

    def _build_pipeline(self) -> Pipeline:
        """
        Builds the Scikit-learn pipeline with preprocessor and classifier.

        The pipeline includes:
        1. Imputation and Scaling for numerical features.
        2. Passthrough for already one-hot encoded categorical features.
        3. Logistic Regression classifier.

        Returns:
            Pipeline: The Scikit-learn pipeline.
        """
        # Define transformer for numerical features: impute missing values then scale
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),  # Replace NaNs with mean
                ("scaler", StandardScaler()),  # Standardize features
            ]
        )

        # Define the column transformer
        # It applies specified transformations to specified columns
        # 'passthrough' means those columns are not transformed but included
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, self.all_numerical_cols_for_pipeline_),
                ("cat_ohe", "passthrough", self.categorical_ohe_cols_),
            ],
            remainder="drop",  # Drop any columns not specified in transformers
            n_jobs=-1,  # Use all available CPU cores for transformers if possible
        )

        # Create the full pipeline: preprocessing then classification
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.logreg_C,
                        solver=self.logreg_solver,
                        max_iter=self.logreg_max_iter,
                        random_state=self.random_state,
                        # n_jobs=-1 can be used for some solvers like 'saga'
                        # For 'lbfgs' (default), it's not directly used for the solver's core computation
                        # but can affect one-vs-rest schemes if applicable.
                    ),
                ),
            ]
        )
        return pipeline

    def _preprocess(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the raw feature DataFrame.

        This method:
        1. Creates date-derived features ('year', 'month', 'day_of_year') from 'Date'.
        2. Ensures numerical features are of numeric type (handling potential errors by coercing to NaN).
        3. Ensures one-hot encoded features are integers (0 or 1).
        4. Selects and orders columns as expected by the pipeline.
        5. Drops the original 'Date' column.

        Args:
            X_raw (pd.DataFrame): Raw features DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame suitable for the model's pipeline.

        Raises:
            ValueError: If essential columns like 'Date' are missing or if other
                        defined features are not found in X_raw after initial processing.
        """
        X_processed = X_raw.copy()

        # Convert 'Date' column to datetime objects and extract features
        if "Date" not in X_processed.columns:
            raise ValueError(
                "Missing 'Date' column in X_raw. Cannot derive date features."
            )

        try:
            X_processed["Date"] = pd.to_datetime(X_processed["Date"])
        except Exception as e:
            raise ValueError(
                f"Error converting 'Date' column to datetime: {e}. Ensure it's in ISO format."
            )

        X_processed["year"] = X_processed["Date"].dt.year
        X_processed["month"] = X_processed["Date"].dt.month
        X_processed["day_of_year"] = X_processed["Date"].dt.dayofyear

        # Convert numerical columns to numeric type, coercing errors to NaN
        # These NaNs will be handled by SimpleImputer in the pipeline
        for col in self.all_numerical_cols_for_pipeline_:
            if col in X_processed.columns:
                X_processed[col] = pd.to_numeric(X_processed[col], errors="coerce")
            else:
                # This check is important for features that should exist (e.g., derived date features)
                # or base numerical/art features if they are missing from X_raw.
                raise ValueError(
                    f"Expected numerical feature '{col}' not found in X_processed columns after date derivation. "
                    f"Original X_raw columns: {list(X_raw.columns)}"
                )

        # Ensure one-hot encoded features are integers (0 or 1)
        for col in self.categorical_ohe_cols_:
            if col in X_processed.columns:
                try:
                    X_processed[col] = X_processed[col].astype(int)
                except ValueError:
                    # If conversion to int fails, try to coerce to numeric then int, filling NaNs with 0
                    # This handles cases where OHE might have non-integer values or NaNs
                    X_processed[col] = (
                        pd.to_numeric(X_processed[col], errors="coerce")
                        .fillna(0)
                        .astype(int)
                    )

                # Validate OHE columns contain only 0 or 1 after conversion
                if not X_processed[col].isin([0, 1]).all():
                    print(
                        f"Warning: Column '{col}' (expected OHE) contains values other than 0 or 1 after processing. "
                        f"Unique values: {X_processed[col].unique()}"
                    )
            else:
                raise ValueError(
                    f"Expected one-hot encoded feature '{col}' not found in X_raw columns: {list(X_raw.columns)}"
                )

        # Drop the original 'Date' column as its information is now in derived features
        if "Date" in X_processed.columns:
            X_processed = X_processed.drop("Date", axis=1)

        # Ensure all columns expected by the pipeline are present and in the correct order
        missing_cols = [
            col
            for col in self.expected_pipeline_input_cols_
            if col not in X_processed.columns
        ]
        if missing_cols:
            raise ValueError(
                f"The following features are missing after preprocessing: {missing_cols}. "
                f"Expected: {self.expected_pipeline_input_cols_}. "
                f"Available: {list(X_processed.columns)}."
            )

        # Return DataFrame with columns in the defined order for the pipeline
        return X_processed[self.expected_pipeline_input_cols_]

    def train(self, X_raw: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the logistic regression model.

        The target `y` (number of guests) is treated as class labels.
        The model pipeline (preprocessing + classifier) is fitted to the provided data.

        Args:
            X_raw (pd.DataFrame): DataFrame with raw features.
            y (pd.Series): Series with the target variable (number of guests).
                           The values in y will be treated as distinct classes.

        Raises:
            TypeError: If X_raw is not a DataFrame or y is not a Series.
            ValueError: If X_raw and y have mismatched number of samples or if y contains NaNs.
        """
        if not isinstance(X_raw, pd.DataFrame):
            raise TypeError("X_raw must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if X_raw.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_raw (shape {X_raw.shape}) and y (shape {y.shape}) "
                "must have the same number of samples."
            )
        if y.isnull().any():
            raise ValueError(
                "Target variable y contains NaN values. Please remove or impute them before training."
            )

        X_processed = self._preprocess(X_raw)
        self.pipeline_ = self._build_pipeline()

        # Fit the pipeline (preprocessing + classifier) to the data
        # NaNs in X_processed numerical columns will be handled by SimpleImputer.
        self.pipeline_.fit(X_processed, y)
        print(
            f"Model training completed. Model uses classes: {self.pipeline_.named_steps['classifier'].classes_}"
        )

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the number of guests using the trained model.

        Args:
            X_raw (pd.DataFrame): DataFrame with raw features for prediction.
                                  Must contain all columns expected by the _preprocess method.

        Returns:
            pd.DataFrame: DataFrame with a single column 'predicted_guests' containing
                          the predicted class labels (number of guests), and using the
                          original index from X_raw.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
            TypeError: If X_raw is not a DataFrame.
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "Model has not been trained or loaded. Call train() or model_load() first."
            )
        if not isinstance(X_raw, pd.DataFrame):
            raise TypeError("X_raw must be a pandas DataFrame.")

        X_processed = self._preprocess(X_raw)

        # Predict using the fitted pipeline
        predictions = self.pipeline_.predict(X_processed)
        return pd.DataFrame({"predicted_guests": predictions}, index=X_raw.index)

    def evaluate(self, X_raw: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the model's performance.

        Performs predictions on X_raw and compares them against y_true.
        Returns a dictionary with accuracy and a classification report.

        Args:
            X_raw (pd.DataFrame): DataFrame with raw features.
            y_true (pd.Series): Series with the true target values.
                                Must have the same index as X_raw for correct alignment.

        Returns:
            Dict[str, Any]: A dictionary containing 'accuracy' (float) and
                            'classification_report' (dict from sklearn.metrics.classification_report).

        Raises:
            RuntimeError: If the model has not been trained or loaded.
            TypeError: If X_raw is not a DataFrame or y_true is not a Series.
            ValueError: If X_raw and y_true have mismatched indices or if y_true contains NaNs.
        """
        if self.pipeline_ is None:
            raise RuntimeError(
                "Model has not been trained or loaded. Call train() or model_load() first."
            )
        if not isinstance(X_raw, pd.DataFrame):
            raise TypeError("X_raw must be a pandas DataFrame.")
        if not isinstance(y_true, pd.Series):
            raise TypeError("y_true must be a pandas Series.")
        if not X_raw.index.equals(y_true.index):
            raise ValueError(
                "X_raw and y_true must have the same index for correct evaluation."
            )
        if y_true.isnull().any():
            raise ValueError("y_true contains NaN values. Evaluation cannot proceed.")

        pred_df = self.predict(X_raw)  # This calls _preprocess internally
        y_pred = pred_df["predicted_guests"]

        accuracy = accuracy_score(y_true, y_pred)

        # Get classes known by the trained model for a comprehensive report
        # Ensure these are strings if y_true or y_pred might be strings, or match types.
        # LogisticRegression.classes_ are usually the same type as y during fit.
        trained_classes = self.pipeline_.named_steps["classifier"].classes_

        # Generate classification report.
        # Using zero_division=0 to avoid warnings/errors for metrics with no support.
        try:
            report = classification_report(
                y_true,
                y_pred,
                labels=trained_classes,
                output_dict=True,
                zero_division=0,
            )
        except ValueError as e:
            # This might happen if y_true contains labels that were not seen during training *and*
            # are also not predicted, leading to issues with `labels` parameter if it restricts evaluation.
            # Fallback: report on all unique labels present in y_true and y_pred.
            print(
                f"Warning during classification_report generation with trained_classes: {e}. "
                "Generating report based on unique labels in y_true and y_pred."
            )
            all_present_labels = sorted(
                list(
                    np.unique(
                        np.concatenate(
                            (y_true.astype(str).unique(), y_pred.astype(str).unique())
                        )
                    )
                )
            )
            report = classification_report(
                y_true,
                y_pred,
                labels=all_present_labels,
                output_dict=True,
                zero_division=0,
            )

        return {"accuracy": accuracy, "classification_report": report}

    def interpret(self, y_pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Interprets the predicted targets.

        For this logistic regression model, the 'predicted_guests' column in the
        input DataFrame (output of `predict` method) directly represents the
        predicted number of guests (as class labels). This method currently
        returns the input DataFrame as is, confirming this direct interpretation.
        Future enhancements could involve adding prediction probabilities, confidence scores, or other interpretations if needed.

        Args:
            y_pred_df (pd.DataFrame): DataFrame with a 'predicted_guests' column,
                                      typically the output of the `predict()` method.

        Returns:
            pd.DataFrame: The input DataFrame, as predictions are already the number of guests.

        Raises:
            TypeError: If y_pred_df is not a pandas DataFrame.
            ValueError: If y_pred_df does not contain a 'predicted_guests' column.
        """
        if not isinstance(y_pred_df, pd.DataFrame):
            raise TypeError("y_pred_df must be a pandas DataFrame.")
        if "predicted_guests" not in y_pred_df.columns:
            raise ValueError("y_pred_df must contain a 'predicted_guests' column.")

        # The prediction 'predicted_guests' is the direct output (class label).
        return y_pred_df

    def model_save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the trained model pipeline (preprocessor + classifier) to a file.

        Uses pickle for serialization. The directory for the path will be created if it doesn't exist.

        Args:
            path (Optional[Union[str, Path]]): Path to save the model.
                                               If None, uses the path defined in `__init__`
                                               (which defaults to `DEFAULT_MODEL_PATH`).
        Raises:
            RuntimeError: If there is no trained model pipeline to save.
            IOError: If there's an issue writing the file.
        """
        save_path = Path(path) if path else self.model_path

        if self.pipeline_ is None:
            raise RuntimeError(
                "No model pipeline to save. Train or load a model first."
            )

        try:
            save_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            with open(save_path, "wb") as f:
                pickle.dump(self.pipeline_, f)
            print(f"Model pipeline saved to {save_path}")
        except IOError as e:
            raise IOError(f"Could not save model to {save_path}: {e}")

    def model_load(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Loads a trained model pipeline (preprocessor + classifier) from a file.

        Uses pickle for deserialization. The loaded pipeline replaces the current `self.pipeline_`.

        Args:
            path (Optional[Union[str, Path]]): Path to load the model from.
                                               If None, uses the path defined in `__init__`
                                               (which defaults to `DEFAULT_MODEL_PATH`).
        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
            TypeError: If the loaded object is not a scikit-learn Pipeline.
            pickle.UnpicklingError: If the file cannot be unpickled.
        """
        load_path = Path(path) if path else self.model_path
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")

        try:
            with open(load_path, "rb") as f:
                loaded_object = pickle.load(f)
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(
                f"Error unpickling model from {load_path}: {e}"
            )
        except IOError as e:  # Catch other potential IO errors during load
            raise IOError(f"Could not load model from {load_path}: {e}")

        if not isinstance(loaded_object, Pipeline):
            raise TypeError(
                f"Loaded object from {load_path} is not a scikit-learn Pipeline. "
                f"Found type: {type(loaded_object)}"
            )

        self.pipeline_ = loaded_object
        print(f"Model pipeline loaded from {load_path}")
        # Note: The hyperparameters (C, solver, etc.) of the loaded model are those
        # it was trained with, which are stored within the pickled LogisticRegression object.
        # The class attributes (self.logreg_C, etc.) are primarily for building *new* models
        # if train() is called on this instance again before loading another model.


if __name__ == "__main__":
    # --- Create dummy data for demonstration ---
    num_samples = 200

    # Generate dates
    dates_raw = pd.date_range(start="2023-01-01", periods=num_samples, freq="D")
    dates_iso = dates_raw.to_series().astype(str)  # Convert to ISO string as expected

    data = {
        "Date": dates_iso,
        # Days of week (simplified, in reality only one would be 1 per row)
        "is_Friday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        "is_Monday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        "is_Saturday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        "is_Sunday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        "is_Thursday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        "is_Tuesday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        "is_Wednesday": np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        # Holidays
        "IsHoliday": np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        "Ascension Day": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "Christmas": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "Day of German Unity": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "Easter Monday": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "Good Friday": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "King's Day": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "May Day": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "New Year's Day": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "Second Christmas Day": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        "Whit Monday": np.random.choice([0, 1], num_samples, p=[0.98, 0.02]),
        # Weather numerical
        "tempmax": np.random.uniform(5, 30, num_samples),
        "tempmin": np.random.uniform(0, 20, num_samples),
        "temp": np.random.uniform(2, 25, num_samples),
        "feelslikemax": np.random.uniform(3, 32, num_samples),
        "feelslikemin": np.random.uniform(-2, 18, num_samples),
        "feelslike": np.random.uniform(0, 28, num_samples),
        "humidity": np.random.uniform(30, 90, num_samples),
        "precip": np.random.uniform(0, 10, num_samples),
        "precipprob": np.random.uniform(0, 100, num_samples),
        "windspeed": np.random.uniform(0, 40, num_samples),
        "cloudcover": np.random.uniform(0, 100, num_samples),
        "solarradiation": np.random.uniform(0, 300, num_samples),
        "uvindex": np.random.uniform(0, 10, num_samples),
        # Weather OHE
        "rain": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        "snow": np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
    }
    # Article ranks (numerical)
    art_names = [
        "art_bloedworst",
        "art_broodplankje",
        "art_captain_dinner",
        "art_carpaccio",
        "art_creme_brulee",
        "art_dame_blanche",
        "art_garnalen_cocktail",
        "art_gehaktbal",
        "art_kaasplankje",
        "art_kalfslever",
        "art_koffie_compleet",
        "art_olijven",
        "art_poffert",
        "art_sate_spies",
        "art_schnitzel",
        "art_sliptong",
        "art_sorbet",
        "art_spareribs",
        "art_stamppot",
        "art_tournedos",
        "art_vers_van_de_markt",
        "art_zalmfilet",
    ]
    for art_col in art_names:
        # Introduce some NaNs for the imputer to handle
        data[art_col] = np.random.choice(
            [0, 1, 2, 3, 4, np.nan], num_samples, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        )

    X_df = pd.DataFrame(data)

    # Target: number of guests (e.g., 10 to 50 guests, in steps of 5)
    # These are treated as discrete classes by Logistic Regression.
    possible_guest_counts = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    y_series = pd.Series(
        np.random.choice(possible_guest_counts, size=num_samples), index=X_df.index
    )

    # --- Split data for train/test demonstration ---
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # --- Create and train model ---
    model = GuestPredictionLogisticModel(
        random_state=42, logreg_max_iter=500
    )  # Increased max_iter for dummy data

    print("\nTraining model...")
    try:
        model.train(X_train, y_train)  # Train on the training data
    except Exception as e:
        print(f"Error during training: {e}")
        try:
            temp_processed_X = model._preprocess(X_train.head())
            print("Sample of preprocessed X for debugging:")
            print(temp_processed_X.head())
            print(temp_processed_X.info())
            print(temp_processed_X.isnull().sum())
        except Exception as pe:
            print(f"Error during debug preprocessing: {pe}")
        raise  # Re-raise the original error

    # --- Make predictions ---
    print(
        "\nMaking predictions on a sample from the TEST set..."
    )  # Predictions on test data
    X_predict_sample = X_test.sample(5, random_state=101)  # Sample from X_test
    predictions_df = model.predict(X_predict_sample)
    print("Predictions (sample from test set):")
    print(predictions_df)
    try:
        print(f"Actual values for this sample:\n{y_test.loc(X_predict_sample.index)}")
    except (KeyError, TypeError):
        print(
            f"Actual values for this sample:\n{y_test.reindex(X_predict_sample.index)}"
        )

    # --- Evaluate model ---
    print(
        "\nEvaluating model on the entire TEST set..."
    )  # Evaluate on the whole test set
    evaluation_results = model.evaluate(
        X_test, y_test
    )  # Use X_test and y_test for evaluation
    print(f"Accuracy on test set: {evaluation_results['accuracy']:.4f}")
    print("Classification Report (on test set):")
    # Print a snippet of the report
    for class_label, metrics in evaluation_results["classification_report"].items():
        # Show metrics for a few actual classes and overall averages
        if str(class_label) in [str(c) for c in possible_guest_counts[:2]] + [
            "macro avg",
            "weighted avg",
        ]:
            print(f"  Class/Avg {class_label}: {metrics}")

    # --- Interpret predictions ---
    print("\nInterpreting predictions...")
    # Use the predictions_df from earlier
    interpreted_predictions = model.interpret(predictions_df)
    print("Interpreted predictions (should be same as predictions DataFrame):")
    print(interpreted_predictions)

    # --- Save model ---
    print("\nSaving model...")
    model_file_path = Path("guest_logistic_predictor_demo.pkl")
    model.model_save(model_file_path)

    # --- Load model ---
    print("\nLoading model...")
    # Create a new instance to simulate loading into a fresh environment
    loaded_model = GuestPredictionLogisticModel(
        random_state=123
    )  # Params here are for new model creation if train() is called
    loaded_model.model_load(model_file_path)

    # --- Predict with loaded model ---
    print("\nMaking predictions with loaded model...")
    # Predict on another small subset from the TEST set
    X_loaded_predict_sample = X_test.sample(3, random_state=103)  # Sample from X_test
    loaded_model_predictions = loaded_model.predict(X_loaded_predict_sample)
    print("Predictions from loaded model (sample from test set):")
    print(loaded_model_predictions)
    try:
        print(
            f"Actual values for this loaded model sample:\n{y_test.loc[X_loaded_predict_sample.index]}"
        )
    except (KeyError, TypeError):
        print(
            f"Actual values for this loaded model sample:\n{y_test.reindex[X_loaded_predict_sample.index]}"
        )

    # --- Clean up dummy model file ---
    if model_file_path.exists():
        model_file_path.unlink()
        print(f"\nCleaned up dummy model file: {model_file_path}")
