import numpy as np

from beartype import beartype


@beartype
def downturn_confidence(
    actual_results: np.ndarray,
    predicted_results: np.ndarray,
) -> tuple[int, int, float] | None:
    """Calculate the downturn confidence."""
    num_correct_downturn_predicted = 0
    num_downturn_predicted = 0
    for actual, predicted in zip(actual_results, predicted_results):
        if predicted == 0:
            num_downturn_predicted += 1
            if predicted == actual:
                num_correct_downturn_predicted += 1
    return (
        None
        if num_downturn_predicted == 0
        else (
            num_correct_downturn_predicted,
            num_downturn_predicted,
            num_correct_downturn_predicted / num_downturn_predicted
        )
    )

@beartype
def print_scores(scores: list[float]) -> None:
    """Print the scores in a formatted way."""
    formatted_scores = [
        "{:.2f}%".format(score * 100)
        for score in scores
    ]
    print(f"[{', '.join(formatted_scores)}]")


@beartype
def print_1_percentage(y: np.ndarray, n_classes: int) -> None:
    percentages = sum(y) / len(y)
    percentages = list(percentages) if n_classes > 1 else [percentages]
    print_scores(percentages)


@beartype
def confusion_matrix_metrics(conf_matrix: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Return the accuracy, precision, recall, specificity, and downturn precision from a confusion matrix.

    Parameters:
    conf_matrix (np.ndarray): A 2x2 numpy array representing the confusion matrix.

    Returns:
    Tuple[float]: A tuple containing the accuracy, precision, recall, specificity, and downturn precision.
    """
    true_negatives = conf_matrix[0][0]
    false_negatives = conf_matrix[1][0]
    true_positives = conf_matrix[1][1]
    false_positives = conf_matrix[0][1]

    accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    downturn_precision = true_negatives / (true_negatives + false_negatives)

    return accuracy, precision, recall, specificity, downturn_precision
