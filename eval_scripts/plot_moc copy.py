import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("./tads/moc.csv")  # your CSV file

plt.figure(figsize=(8,6))
sns.boxplot(x='tool', y='moc', data=df, palette="Set2")
sns.swarmplot(x='tool', y='moc', data=df, color=".25")  # optional: show individual points
plt.ylabel("MOC")
plt.title("Distribution of MOC scores per tool")
plt.savefig("moc_boxplot.png")

plt.figure(figsize=(8,6))
sns.violinplot(x='tool', y='moc', data=df, palette="Set2", inner="quartile")
plt.ylabel("MOC")
plt.title("MOC score distribution per tool")
plt.savefig("moc_violinplot.png")

heat_df = df.pivot_table(index=['dataset', 'chromosome'], columns='tool', values='moc')
plt.figure(figsize=(8,6))
sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("MOC scores heatmap")
plt.savefig("moc_heatmap.png")

subset = df[df['tool'].isin(['EmbedTAD','Spectral'])]
pivot = subset.pivot(index=['dataset','chromosome'], columns='tool', values='moc')
plt.scatter(pivot['EmbedTAD'], pivot['Spectral'])
plt.plot([0,100],[0,100], 'r--')
plt.xlabel("EmbedTAD MOC")
plt.ylabel("Spectral MOC")
plt.title("Tool consistency comparison")
plt.savefig("moc_scatter.png")
