import os
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

FID = [87.51, 50.21, 41.71, 20.78, 11.03, 7.09, 5.22, 4.43, 4.13][::-1]
ACC = [62.08, 60.52, 62.18, 62.91, 64.65, 66.56, 66.68, 67.63, 67.86][::-1]

os.makedirs("output/paper_plots", exist_ok=True)

# Plot a grid
fig = plt.figure(figsize=(5.5, 3))
fontsize = 14

# Draw curves
plt.plot(FID, ACC, linewidth=0.75, marker='x', color="#0071BC", label="Squeeze and span")
plt.plot([4.13], [65.22], linewidth=0.75, marker='*', color="magenta", label="VICReg on mixture")
plt.axhline(63.31, linewidth=0.75, linestyle='-.', color="black", label="VICReg on real")

# Stylize
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.xscale('log')
plt.xticks([87.51, 41.71, 20.78, 11.03, 7.09, 5.22, 4.13])
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xlabel("FID (log-scale)", fontsize=fontsize)
plt.ylabel('Top1 Acc (%)', fontsize=fontsize)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(loc="upper right", ncol=1)
fig.savefig(f"output/paper_plots/acc_vs_fid.pdf", bbox_inches='tight')
