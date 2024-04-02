import matplotlib.pyplot as plt


def scatter_plot(df):
    colors = plt.cm.tab20.colors[:len(df.columns)]
    for column_name, color in zip(df, colors):
        plt.scatter(df[column_name].index.values, df[column_name].values, color=color, label=column_name)
    plt.legend()
    plt.show()