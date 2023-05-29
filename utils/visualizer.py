import matplotlib.pyplot as plt


def plot(model, test_loader, X_train, Y_train, X_test, Y_test, title=''):
    """
    Plot model prediction along with training data.
    """
    fig = plt.figure(figsize=(12, 4))
    plt.title(title, fontsize=11)

    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        model.test()
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model.pred(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
    means = means[1:]
    variances = variances[1:]

    plt.plot(X_train, Y_train, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(X_test, means, lw=2.5, label="Mean of predictive posterior")
    col = line.get_color()
    plt.fill_between(
        X_test[:, 0],
        (means - 2 * variances ** 0.5),
        (means + 2 * variances ** 0.5),
        color=col,
        alpha=0.2,
        lw=1.5,
    )
    Z = model.get_inducing_points().cpu().numpy()
    plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
    plt.legend(loc="lower right", fontsize=11)
    # plt.show()
    mae = torch.mean(torch.abs(means - Y_test.cpu()))
    neptune_run['evaluation/plot/'+title].upload(fig)
    neptune_run['training/epoch/mae'].log(mae)
    plt.close(fig)