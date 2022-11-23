import matplotlib.pyplot as plt
import numpy as np

# Convert square units to decibels
def db(x):
    return 10 * np.log10(x)

# Show final model metrics and plot trainig curves
def show_model_metrics(metrics, modelname):
    # Show model metrics
    print(modelname + ' Metrics:')
    print('Train Loss: {0:.5f}'.format(metrics.loss_train[-1]))
    print('Test Loss : {0:.5f}'.format(metrics.loss_test[-1]))
    print('Train Accuracy: {0:.2f} %'.format(metrics.accuracy_train[-1]))
    print('Test Accuracy : {0:.2f} %'.format(metrics.accuracy_test[-1]))

    # Initialize multiple plot figure
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))

    # Plot loss curves
    epochs = np.arange(1, len(metrics.loss_train) + 1)
    ax[0].plot(epochs, db(metrics.loss_train), label='Train set')
    ax[0].plot(epochs, db(metrics.loss_test), label='Test set')

    # Plot accuracy curves
    ax[1].plot(epochs, metrics.accuracy_train)
    ax[1].plot(epochs, metrics.accuracy_test)

    # Set axis limits
    ax[0].set_xlim([0, epochs[-1]])
    ax[1].set_xlim([0, epochs[-1]])

    # Set plot titles and labels
    ax[0].set_xlabel("Epoch Number")
    ax[0].set_ylabel("Loss (dB)")
    ax[1].set_xlabel("Epoch Number")
    ax[1].set_ylabel("Accuracy (%)")
    ax[0].set_title("Loss Curves")
    ax[1].set_title("Accuracy Curves")

    # Setup parent figure
    fig.suptitle(modelname + ' Training Results', y=1.04)
    fig.subplots_adjust(hspace=1.5, wspace=0.4)
    fig.legend(bbox_to_anchor=(0.67, -0.2), loc='lower right', ncol=4)
    plt.show()