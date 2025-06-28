import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_digits_data():
    digits = datasets.load_digits()
    return digits

def visualize_samples(images, labels, num=4, title_prefix="Training"):
    _, axes = plt.subplots(nrows=1, ncols=num, figsize=(10, 3))
    for ax, image, label in zip(axes, images, labels):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{title_prefix}: {label}")
    plt.show()

def preprocess_data(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data

def split_data(data, target):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.5, shuffle=False
    )
    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train):
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf

def predict(clf, X_test):
    return clf.predict(X_test)

def evaluate_and_plot(clf, X_test, y_test, predicted):
    # Visualize predictions
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.show()

    # Print classification report
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    # Confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    # Rebuild report from confusion matrix
    cm = disp.confusion_matrix
    y_true = []
    y_pred = []
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )
