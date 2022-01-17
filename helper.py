import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, correlationFood, correlationDanger):
    plt.clf()

    plt.title('Scores')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.grid()

    plt.show()
    plt.pause(.1)