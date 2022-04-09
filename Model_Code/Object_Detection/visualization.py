### Visualizations ###

### External Imports ###
import csv
import numpy as np
import matplotlib.pyplot as plt 
### External Imports ###

def displayImage():
    pass 
# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# # image shape is [batch_size, 3 (due to RGB), height, width]
# print(train_features.shape)
# img = transforms.ToPILImage()(train_features[0])
# plt.imshow(img)
# plt.show()
# #print(train_labels)

# Would be cool to have a function that plotted losses
def plotLoses(stats_file):
    stats = open(stats_file, 'r')
    read_stats = csv.DictReader(stats)

    losses = np.empty(0)
    step = np.empty(0)

    i = 1
    for lines in read_stats:
        losses = np.append(losses, float(lines['Loss']))
        step = np.append(step, (int(lines['Epoch']) * i))
        i += 1

    y = np.arange(start=1, stop=(len(losses)+1), step=1)

    plt.title("Loss Plot")
    plt.plot(y, losses, color="red")
    plt.show()

plotLoses('train_stats00.csv')

### Visualizations ###