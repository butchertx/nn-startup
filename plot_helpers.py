import matplotlib.pyplot as plt

def plot_2_lines(x_data, 
                y1_data, y2_data,
                y1_label='y1', y2_label='y2', 
                x_label='xlabel', y_label='ylabel', title='title') :
    plt.clf()
    plt.plot(x_data, y1_data, 'bo', label=y1_label)
    plt.plot(x_data, y2_data, 'b', label=y2_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

