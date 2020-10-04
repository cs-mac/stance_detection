import seaborn as sn
from matplotlib import pyplot as plt
from collections import Counter

def create_confusion_matrix(confusion_m, categories, y_lim_value=5, title="cm", show_plots=False, save_plots=False, method="TRAINING", fig_size=(16,9)):
    '''
    Creates a confusion matrix
    '''
    plt.figure(figsize = fig_size, dpi=150)
    sn.set(font_scale=2.5) #label size
    hm = sn.heatmap(confusion_m, annot=True, fmt='g', annot_kws={"size": 32}) #font size
    hm.set_ylim(y_lim_value, 0)
    hm.set(xticklabels = categories, yticklabels = categories)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plt.title(title + ' Confusion Matrix')
    if show_plots:
        plt.show()
    if save_plots:
        hm.figure.savefig("./results/" + method + "_" + title + '_CM' + '.png', figsize = (16, 9), dpi=150, bbox_inches="tight")
    plt.close()


def print_distribution(l):
    '''
    Get distribution from list
    '''
    print(Counter(l)) 
