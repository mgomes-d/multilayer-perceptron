import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(df):

    features_subset = df[df.columns[21:]].copy()
    # features_subset.columns += 'Diagnosis'
    features_subset['Diagnosis'] = df['Diagnosis']
    # print(features_subset)
    pair_plot = sns.pairplot(features_subset, hue='Diagnosis')
    plt.show()
