def train_linear_regression(x, y):
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n

    b1 = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) / \
         sum((x[i] - x_mean) ** 2 for i in range(n))
    b0 = y_mean - b1 * x_mean

    return b0, b1


if __name__ == "__main__":
    x = list(map(float, input("Enter X values (space separated): ").split()))
    y = list(map(float, input("Enter Y values (space separated): ").split()))

    b0, b1 = train_linear_regression(x, y)
    print(f"Model: y = {b0:.4f} + {b1:.4f}x")

    x_new = float(input("Enter X value to predict Y: "))
    print("Predicted Y:", round(b0 + b1 * x_new, 4))
