from src.logger import logger
from src.exception import CustomException
from src.utils import load_object

import os
import pandas as pd
from dataclasses import dataclass


@dataclass
class PredictionPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()

    def predict(self, features: pd.DataFrame) -> list[float]:
        """
        Perform predictions using a pre-trained model and preprocessor.
        This method takes a DataFrame of input features, preprocesses them using a
        pre-trained preprocessor, and then generates predictions using a pre-trained model.

        Args:
            features (pd.DataFrame): A DataFrame containing the input features for prediction.

        Raises:
            CustomException: If an error occurs during the prediction process, such as
                             issues with loading the model/preprocessor or during preprocessing.

        Returns:
            list[float]: The predicted values. Returns a single float in list if the input
                                 corresponds to a single instance, or a list of floats for
                                 multiple instances.
        """

        logger.info("Prediction pipeline started")
        try:
            logger.info("Loading model and preprocessor")
            model = load_object(self.config.model_path)
            preprocessor = load_object(self.config.preprocessor_path)

            logger.info("Preprocessing features")
            processed_features = preprocessor.transform(features)

            logger.info("Making predictions")
            preds = model.predict(processed_features)

            logger.info("Prediction pipeline completed")

            return preds
        except Exception as e:
            raise CustomException(e)


@dataclass
class StudentExamRecord:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def __post_init__(self):
        # Define valid categories
        valid_genders = {"male", "female"}
        valid_race_ethnicities = {"group A", "group B", "group C", "group D", "group E"}
        valid_parent_education = {
            "bachelor's degree",
            "some college",
            "master's degree",
            "associate's degree",
            "high school",
            "some high school",
        }
        valid_lunch_types = {"standard", "free/reduced"}
        valid_test_prep_courses = {"none", "completed"}

        # Validate gender
        if self.gender not in valid_genders:
            raise ValueError(
                f"Invalid gender '{self.gender}'. Must be one of {valid_genders}."
            )

        # Validate race_ethnicity
        if self.race_ethnicity not in valid_race_ethnicities:
            raise ValueError(
                f"Invalid race_ethnicity '{self.race_ethnicity}'. Must be one of {valid_race_ethnicities}."
            )

        # Validate parental level of education
        if self.parental_level_of_education not in valid_parent_education:
            raise ValueError(
                f"Invalid parental_level_of_education '{self.parental_level_of_education}'. Must be one of {valid_parent_education}."
            )

        # Validate lunch type
        if self.lunch not in valid_lunch_types:
            raise ValueError(
                f"Invalid lunch type '{self.lunch}'. Must be one of {valid_lunch_types}."
            )

        # Validate test preparation course
        if self.test_preparation_course not in valid_test_prep_courses:
            raise ValueError(
                f"Invalid test_preparation_course '{self.test_preparation_course}'. Must be one of {valid_test_prep_courses}."
            )

        # Validate scores (must be between 0 and 100)
        if not (0 <= self.reading_score <= 100):
            raise ValueError(
                f"Invalid reading_score '{self.reading_score}'. Must be between 0 and 100."
            )

        if not (0 <= self.writing_score <= 100):
            raise ValueError(
                f"Invalid writing_score '{self.writing_score}'. Must be between 0 and 100."
            )


class CustomData:
    def __init__(self, records: list[StudentExamRecord]):
        # Ensure that data is a list of CustomDataModel instances
        if not all(isinstance(entry, StudentExamRecord) for entry in records):
            raise CustomException("Invalid data format")

        self.df = pd.DataFrame(records)


if __name__ == "__main__":
    pipeline = PredictionPipeline()

    students_data = CustomData(
        records=[
            StudentExamRecord(
                gender="male",
                race_ethnicity="group B",
                parental_level_of_education="some college",
                lunch="standard",
                test_preparation_course="none",
                reading_score=72,
                writing_score=83,
            ),
            StudentExamRecord(
                gender="female",
                race_ethnicity="group C",
                parental_level_of_education="bachelor's degree",
                lunch="free/reduced",
                test_preparation_course="completed",
                reading_score=88,
                writing_score=92,
            ),
        ]
    )

    predictions = pipeline.predict(students_data.df)
    print(predictions)
