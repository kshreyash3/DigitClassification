from utils import (
    load_digits_data,
    visualize_samples,
    preprocess_data,
    split_data,
    train_classifier,
    predict,
    evaluate_and_plot,
)

def main():
    digits = load_digits_data()
    visualize_samples(digits.images, digits.target, num=4, title_prefix="Training")

    data = preprocess_data(digits)
    X_train, X_test, y_train, y_test = split_data(data, digits.target)

    clf = train_classifier(X_train, y_train)
    predicted = predict(clf, X_test)

    evaluate_and_plot(clf, X_test, y_test, predicted)

if __name__ == "__main__":
    main()
