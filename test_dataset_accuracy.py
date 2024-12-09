import pandas as pd
from collections import Counter
import numpy as np

def calculate_accuracy(pred, actual):
    correct_predictions = sum(p == a for p, a in zip(pred, actual))
    accuracy = correct_predictions / len(actual)
    return accuracy

def calculate_classwise_accuracy(pred, actual):
    class_wise_correct = Counter()
    class_wise_total = Counter()

    for p, a in zip(pred, actual):
        class_wise_total[a] += 1  # Count total instances of each class
        if p == a:
            class_wise_correct[a] += 1  # Count correct predictions for each class

    classwise_accuracy = {cls: class_wise_correct[cls] / class_wise_total[cls]
                          for cls in class_wise_total}
    return classwise_accuracy


def analyze_wrong_predictions(pred, actual):
    wrong_predictions = []
    for i, (p, a) in enumerate(zip(pred, actual)):
        if p != a:
            wrong_predictions.append((i, p, a))  # Store index, predicted class, and actual class
    return wrong_predictions


if __name__ == '__main__':

    path_to_test_dataset_corrections = '/4TBHD/ISL/CodeBase/Test_Dataset_Corrections/right_elbow_orientation.csv'

    df = pd.read_csv(path_to_test_dataset_corrections)

    pred , actual = [],[]
    for index, row in df.iterrows():
        prediction_value = row['Predictions']
        correction_value = row['Corrections']

        if pd.notna(prediction_value) and pd.notna(correction_value):

            pred.append(prediction_value)
            actual.append(correction_value)
    

accuracy = calculate_accuracy(pred, actual)
classwise_accuracy = calculate_classwise_accuracy(pred, actual)
# wrong_predictions = analyze_wrong_predictions(pred, actual)

print("Overall Accuracy:", accuracy)
print("Class-wise Accuracy:", classwise_accuracy)
# print("Wrong Predictions (Index, Predicted, Actual):", wrong_predictions)
