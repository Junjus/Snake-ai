import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, correlationFood, correlationDanger):
    plt.clf()

    plt.subplot(2,1,1)
    plt.title('Scores')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.grid()

    plt.subplot(2,1,2)
    plt.title('Correlations')
    plt.xlabel('Number of Games')
    plt.ylabel('Correlation Coefficient')
    plt.plot(correlationFood, color='c')
    plt.plot(correlationDanger, color='m')
    plt.ylim(ymin=-1, ymax=1)
    plt.xlim(xmin=0)
    plt.grid()

    plt.show()
    plt.pause(.1)