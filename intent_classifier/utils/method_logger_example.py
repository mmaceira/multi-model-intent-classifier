"""
Simple example usage of the Method Logger.

This demonstrates how to use the logger to track method inputs, outputs, and errors.
"""

from intent_classifier.utils.method_logger import get_logger, log_method


# Example 1: Using the decorator (easiest way)
class MyClassifier:
    """Example classifier with automatic logging."""

    @log_method
    def predict(self, docs):
        """This method will automatically log inputs, outputs, and errors."""
        results = []
        for doc in docs:
            # Your prediction logic here
            result = f"predicted_label_for_{doc[:10]}"
            results.append(result)
        return results

    @log_method
    def predict_proba(self, docs):
        """This will also be logged automatically."""
        # Your probability calculation here
        return {"label1": 0.8, "label2": 0.2}


# Example 2: Manual logging for predictions
def example_manual_logging():
    """Example of manually logging predictions."""
    logger = get_logger()

    def predict_with_logging(docs):
        """Predict and log each prediction."""
        results = []
        for doc in docs:
            try:
                # Your prediction logic
                prediction = "some_label"

                # Log the prediction
                logger.log_prediction(
                    input=doc,
                    output=prediction,
                    metadata={"model": "example_model"},
                )

                results.append(prediction)
            except Exception as e:
                # Errors are automatically logged if using @log_method
                # Or log manually:
                logger.log_error(
                    method_name="predict_with_logging",
                    error=e,
                    inputs={"doc": doc},
                )
                raise

        return results


# Example 3: Integration in existing code
# Just add @log_method decorator to any method you want to log:


class ExistingClass:
    @log_method
    def existing_method(self, param1, param2):
        """This method will now be logged automatically."""
        return param1 + param2


# Example 4: Using in adapter_sklearn.py
# You would modify adapter_sklearn.py like this:
"""
from intent_classifier.utils.method_logger import log_method

class RagSklearnAdapter(BaseEstimator, ClassifierMixin):
    @log_method
    def predict(self, X):
        # This will automatically log inputs (X) and outputs (predictions)
        if isinstance(X, list):
            return self.rag.predict(X)
        ...

    @log_method
    def predict_proba(self, X):
        # This will also be logged
        ...
"""


# Example 5: Using in rag_llm/classifier.py
# You would modify classifier.py like this:
"""
from intent_classifier.utils.method_logger import log_method, get_logger

class RagLLM(RagClassifierBase):
    @log_method
    def predict(self, docs: Sequence[str], **kwargs) -> List[str]:
        logger = get_logger()
        results = []
        for doc in docs:
            try:
                result = classify_single(...)
                # Log each prediction
                logger.log_prediction(
                    input=doc,
                    output=result["label"],
                )
                results.append(result["label"])
            except Exception as e:
                # Error is automatically logged by @log_method
                results.append(self.labels[0] if self.labels else "unknown")
        return results
"""


if __name__ == "__main__":
    # Initialize logger
    logger = get_logger()
    print(f"Logger initialized. Log directory: {logger.log_dir}")

    # Example usage
    classifier = MyClassifier()
    results = classifier.predict(["test document 1", "test document 2"])
    print(f"Predictions: {results}")

    print("\nLogging examples completed. Check the log directory for output files:")
    print("  - method_calls_YYYYMMDD.jsonl (method inputs and outputs)")
    print("  - predictions_YYYYMMDD.jsonl (prediction logs)")
    print("  - errors_YYYYMMDD.jsonl (error logs)")
