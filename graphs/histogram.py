import matplotlib.pyplot as plt

def histogram(df):
    num_row = 6
    num_col = 5
    num_graphs = len(df.columns)
    fig, axes = plt.subplots(num_row, num_col, figsize=(12,8))

    axes_flatten = axes.flatten()
    colors_tab20 = plt.cm.tab20.colors
    colors = colors_tab20 * 2 

    for i, (ax, column_name) in enumerate(zip(axes_flatten, df.columns)):
        df[column_name].plot.hist(ax=ax, color=colors[i], edgecolor='black')
        ax.set_title(f'{column_name}', fontsize = 12)
    plt.tight_layout()
    plt.show()
