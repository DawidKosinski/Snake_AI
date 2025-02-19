import matplotlib.pyplot as plt
from IPython import display


plt.ion()

def plot(scores, avg_score):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    plt.clf()
    
    plt.title('Training Progress')
    
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores)
    plt.plot(avg_score)
    
    plt.ylim(ymin=0)
    
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(avg_score)-1, avg_score[-1], str(avg_score[-1]))
    
    plt.show(block=False)
    plt.pause(.1)
