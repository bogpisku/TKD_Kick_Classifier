import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

features= pd.read_csv("../DataSets/50_kicks/linear_acceleration/full_dataset.csv")

classes = ["Y","A","P","S"]

for i in range(len(features)):
    print("Changing Class Names")
    features.iloc[i, -1] = classes[int(features.iloc[i, -1])] #change int of the class for the letter
    # cat = features.iloc[i,-1]
    # print(cat)

    # if cat == 0:
    #     features.iloc[i,-1] = "Squat"
    # elif cat == 1:
    #     features.iloc[i,-1] = "Walk"
    # elif cat == 2:
    #     features.iloc[i,-1] = "Run"

###############################################################################
print("Generating Mean Acceleration Values Plots")
fig = plt.figure(figsize=(18, 11))
plt.subplot(321)
sns.scatterplot(x='Mx', y="Mm", hue='Class', data=features)
plt.subplot(323)
sns.scatterplot(x='My', y="Mm", hue='Class', data=features)
plt.subplot(325)
sns.scatterplot(x='Mz', y="Mm", hue='Class', data=features)
plt.subplot(322)
sns.scatterplot(x='Mx', y="My", hue='Class', data=features)
plt.subplot(324)
sns.scatterplot(x='Mx', y="Mz", hue='Class', data=features)
plt.subplot(326)
sns.scatterplot(x='My', y="Mz", hue='Class', data=features)
fig.suptitle("Mean Acceleration Values", fontsize=30)
plt.savefig("plots/meanAcc.png", bbox_inches="tight")
# ###############################################################################
print("Generating Average Peak Frequency Plots")
fig = plt.figure(figsize=(18, 11))
plt.subplot(321)
sns.scatterplot(x='APFx', y="APFm", hue='Class', data=features)
plt.subplot(323)
sns.scatterplot(x='APFy', y="APFm", hue='Class', data=features)
plt.subplot(325)
sns.scatterplot(x='APFz', y="APFm", hue='Class', data=features)
plt.subplot(322)
sns.scatterplot(x='APFx', y="APFy", hue='Class', data=features)
plt.subplot(324)
sns.scatterplot(x='APFx', y="APFz", hue='Class', data=features)
plt.subplot(326)
sns.scatterplot(x='APFy', y="APFz", hue='Class', data=features)
fig.suptitle("Average Peak Frequency", fontsize=30)
plt.savefig("plots/apf.png", bbox_inches="tight")
# ###############################################################################
print("Generating Root Mean Square Value Plots")
fig = plt.figure(figsize=(18, 11))
plt.subplot(321)
sns.scatterplot(x='RMSx', y="RMSm", hue='Class', data=features)
plt.subplot(323)
sns.scatterplot(x='RMSy', y="RMSm", hue='Class', data=features)
plt.subplot(325)
sns.scatterplot(x='RMSz', y="RMSm", hue='Class', data=features)
plt.subplot(322)
sns.scatterplot(x='RMSx', y="RMSy", hue='Class', data=features)
plt.subplot(324)
sns.scatterplot(x='RMSx', y="RMSz", hue='Class', data=features)
plt.subplot(326)
sns.scatterplot(x='RMSy', y="RMSz", hue='Class', data=features)
fig.suptitle("Root Mean Square Value", fontsize=30)
plt.savefig("plots/rms.png", bbox_inches="tight")
# ###############################################################################
print("Generating Standard Deviation Plots")
fig = plt.figure(figsize=(18, 11))
plt.subplot(321)
sns.scatterplot(x='SDx', y="SDm", hue='Class', data=features)
plt.subplot(323)
sns.scatterplot(x='SDy', y="SDm", hue='Class', data=features)
plt.subplot(325)
sns.scatterplot(x='SDz', y="SDm", hue='Class', data=features)
plt.subplot(322)
sns.scatterplot(x='SDx', y="SDy", hue='Class', data=features)
plt.subplot(324)
sns.scatterplot(x='SDx', y="SDz", hue='Class', data=features)
plt.subplot(326)
sns.scatterplot(x='SDy', y="SDz", hue='Class', data=features)
fig.suptitle("Standard Deviation", fontsize=30)
plt.savefig("plots/std.png", bbox_inches="tight")
# ###############################################################################
print("Generating Min-Max Difference Plots")
fig = plt.figure(figsize=(18, 11))
plt.subplot(321)
sns.scatterplot(x='MMx', y="MMm", hue='Class', data=features)
plt.subplot(323)
sns.scatterplot(x='MMy', y="MMm", hue='Class', data=features)
plt.subplot(325)
sns.scatterplot(x='MMz', y="MMm", hue='Class', data=features)
plt.subplot(322)
sns.scatterplot(x='MMx', y="MMy", hue='Class', data=features)
plt.subplot(324)
sns.scatterplot(x='MMx', y="MMz", hue='Class', data=features)
plt.subplot(326)
sns.scatterplot(x='MMy', y="MMz", hue='Class', data=features)
fig.suptitle("Min-Max Difference", fontsize=30)
plt.savefig("plots/minmax.png", bbox_inches="tight")
# ###############################################################################
print("Generating Correlation between axes Plots")
fig = plt.figure(figsize=(6, 11))
plt.subplot(311)
sns.scatterplot(x='Cxy', y="Cxz", hue='Class', data=features)
plt.subplot(312)
sns.scatterplot(x='Cxy', y="Cyz", hue='Class', data=features)
plt.subplot(313)
sns.scatterplot(x='Cxz', y="Cyz", hue='Class', data=features)
fig.suptitle("Correlation between axes", fontsize=30)
plt.savefig("plots/corr.png", bbox_inches="tight")
# ##############################################################
print("Generating APF variance Plots")
fig = plt.figure(figsize=(11, 6))
g = sns.FacetGrid(features, col='Class')
g = g.map(sns.kdeplot, 'varAPF')
fig.suptitle("APF variance", fontsize=30)
plt.savefig("plots/varAPF.png", bbox_inches="tight")