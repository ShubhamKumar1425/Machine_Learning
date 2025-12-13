import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n = len(X)
    b0 = 0.0  # intercept
    b1 = 0.0  # slope

    for _ in range(epochs):
        y_pred = b0 + b1 * X
        error = y_pred - y

        db0 = (1 / n) * np.sum(error)
        db1 = (1 / n) * np.sum(error * X)

        b0 -= lr * db0
        b1 -= lr * db1

    return b0, b1


if __name__ == "__main__":
    X = np.array(list(map(float, input("Enter X values (space separated): ").split())))
    y = np.array(list(map(float, input("Enter Y values (space separated): ").split())))

    lr = float(input("Enter learning rate (e.g. 0.01): "))
    epochs = int(input("Enter number of iterations: "))

    b0, b1 = gradient_descent(X, y, lr, epochs)

    print(f"\nFinal Model: y = {b0:.4f} + {b1:.4f}x")

    x_new = float(input("Enter X value to predict Y: "))
    print("Predicted Y:", round(b0 + b1 * x_new, 4))
