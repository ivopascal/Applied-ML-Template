import pickle
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression  # Changed to LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)  # Changed for regression metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class GuestPredictionLinearModel:  # Renamed class
    """
    A linear regression model to predict the number of guests (as a continuous value).

    This class encapsulates preprocessing, training, prediction, evaluation,
    and model persistence functionalities. It uses a Scikit-learn pipeline
    to handle feature transformations and the linear regression estimator.

    Raw Features Expected:
    - Date: date (ISO string)
    - is_Friday, is_Monday, ..., is_Wednesday: one-hot encoded (0/1)
    - IsHoliday: one-hot encoded (0/1)
    - Ascension Day, Christmas, ..., Whit Monday: one-hot encoded (0/1)
    - tempmax, tempmin, ..., uvindex: numerical weather data
    - rain, snow: one-hot encoded (0/1) weather conditions
    - art_bloedworst, ..., art_zalmfilet: numerical rank of articles

    Target:
    - Number of guests (treated as a continuous numerical value for linear regression).
    """

    DEFAULT_MODEL_PATH: Path = Path("guests_linear_model.pkl")

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        fit_intercept: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Initializes the GuestPredictionLinearModel.

        Args:
            model_path (Optional[Union[str, Path]]): Path to load/save the model.
                                                     Defaults to DEFAULT_MODEL_PATH.
            fit_intercept (bool): Whether to calculate the intercept for this model.
                                  If set to False, no intercept will be used in calculations
                                  (e.g., data is expected to be already centered).
            random_state (Optional[int]): Controls the randomness of the model if applicable.
                                          LinearRegression itself is deterministic.
        """
        self.model_path: Path = (
            Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        )
        self.pipeline_: Optional[Pipeline] = None  # Populated after training or loading
        self.fit_intercept: bool = fit_intercept
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
        # self.art_cols_: List[str] = [
        #     "art_bloedworst",
        #     "art_broodplankje",
        #     "art_captain_dinner",
        #     "art_carpaccio",
        #     "art_creme_brulee",
        #     "art_dame_blanche",
        #     "art_garnalen_cocktail",
        #     "art_gehaktbal",
        #     "art_kaasplankje",
        #     "art_kalfslever",
        #     "art_koffie_compleet",
        #     "art_olijven",
        #     "art_poffert",
        #     "art_sate_spies",
        #     "art_schnitzel",
        #     "art_sliptong",
        #     "art_sorbet",
        #     "art_spareribs",
        #     "art_stamppot",
        #     "art_tournedos",
        #     "art_vers_van_de_markt",
        #     "art_zalmfilet",
        # ]

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
        Builds the Scikit-learn pipeline with preprocessor and regressor.

        The pipeline includes:
        1. Imputation and Scaling for numerical features.
        2. Passthrough for already one-hot encoded categorical features.
        3. Linear Regression regressor.

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

        # Create the full pipeline: preprocessing then regression
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    LinearRegression(
                        fit_intercept=self.fit_intercept,
                        n_jobs=-1,  # Use all available CPU cores for fitting if possible
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
        Trains the linear regression model.

        The target `y` (number of guests) is treated as a continuous numerical value.
        The model pipeline (preprocessing + regressor) is fitted to the provided data.

        Args:
            X_raw (pd.DataFrame): DataFrame with raw features.
            y (pd.Series): Series with the target variable (number of guests).
                           The values in y will be treated as continuous.

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

        # Fit the pipeline (preprocessing + regressor) to the data
        # NaNs in X_processed numerical columns will be handled by SimpleImputer.
        self.pipeline_.fit(X_processed, y)
        print("Model training completed.")

    def predict(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the number of guests using the trained model.

        Args:
            X_raw (pd.DataFrame): DataFrame with raw features for prediction.
                                  Must contain all columns expected by the _preprocess method.

        Returns:
            pd.DataFrame: DataFrame with a single column 'predicted_guests' containing
                          the predicted continuous numerical values for the number of guests,
                          and using the original index from X_raw.

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
        # Ensure predictions are non-negative for guest counts, and potentially rounded if desired.
        # For a linear model, predictions can be negative, so clipping is often useful.
        predictions = np.maximum(0, predictions).round().astype(int)
        return pd.DataFrame({"predicted_guests": predictions}, index=X_raw.index)

    def evaluate(self, X_raw: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the model's performance using regression metrics.

        Performs predictions on X_raw and compares them against y_true.
        Returns a dictionary with Mean Absolute Error (MAE) and R-squared (R2) score.

        Args:
            X_raw (pd.DataFrame): DataFrame with raw features.
            y_true (pd.Series): Series with the true target values.
                                Must have the same index as X_raw for correct alignment.

        Returns:
            Dict[str, Any]: A dictionary containing 'mean_absolute_error' (float) and
                            'r2_score' (float).

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

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"mean_absolute_error": mae, "r2_score": r2}

    def interpret(self, y_pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Interprets the predicted targets.

        For this linear regression model, the 'predicted_guests' column in the
        input DataFrame directly represents the predicted continuous number of guests.
        This method currently returns the input DataFrame as is, confirming this direct interpretation.

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

        # The prediction 'predicted_guests' is the direct output (continuous value).
        return y_pred_df

    def model_save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the trained model pipeline (preprocessor + regressor) to a file.

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
        Loads a trained model pipeline (preprocessor + regressor) from a file.

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


if __name__ == "__main__":
    Xy_df = pd.read_csv("../../../data/cleaned/full_restaurant_data_rank_int.csv", header=0)

    X_df = Xy_df.copy().drop(["GUESTS"], axis=1)

    y_series = pd.Series(Xy_df["GUESTS"])

    # --- Split data for train/test demonstration ---
    print("\nSplitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # --- Create and train model ---
    model = GuestPredictionLinearModel(random_state=42)

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
    print(
        f"Mean Absolute Error on test set (Linear Model): {evaluation_results['mean_absolute_error']:.4f}"
    )
    print(
        f"R-squared (R2) score on test set (Linear Model): {evaluation_results['r2_score']:.4f}"
    )

    # --- Interpret predictions ---
    print("\nInterpreting predictions...")
    # Use the predictions_df from earlier
    interpreted_predictions = model.interpret(predictions_df)
    print("Interpreted predictions (should be same as predictions DataFrame):")
    print(interpreted_predictions)

    # --- Save model ---
    print("\nSaving model...")
    model_file_path = Path("guest_linear_predictor_demo.pkl")
    model.model_save(model_file_path)

    # --- Load model ---
    print("\nLoading model...")
    loaded_model = GuestPredictionLinearModel(
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
