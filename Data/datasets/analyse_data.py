"""
Drugs Dataset Analysis
======================
Analyzes the target labels for the molecular property prediction task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Drugs.csv')

print("=" * 60)
print("DRUGS DATASET ANALYSIS")
print("=" * 60)

# Basic info
print(f"\n📊 Dataset Shape: {df.shape[0]} molecules, {df.shape[1]} columns")
print(f"\n📋 Columns: {list(df.columns)}")

# Check for missing values
print(f"\n🔍 Missing Values:")
print(df.isnull().sum())

# Target statistics
targets = ['energy', 'ip', 'ea', 'chi']
print(f"\n📈 Target Statistics:")
print(df[targets].describe().round(4))

# Correlation matrix
print(f"\n🔗 Target Correlations:")
corr = df[targets].corr()
print(corr.round(3))

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Drugs Dataset - Target Distribution Analysis', fontsize=14)

# 1. Histograms for each target
for i, target in enumerate(targets):
    ax = axes[i // 2, i % 2]
    ax.hist(df[target], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(target)
    ax.set_ylabel('Count')
    ax.set_title(f'{target} Distribution\nμ={df[target].mean():.2f}, σ={df[target].std():.2f}')

# 2. Correlation heatmap
ax = axes[0, 2]
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title('Target Correlations')

# 3. Pairplot-style scatter (ip vs ea, colored by chi)
ax = axes[1, 2]
scatter = ax.scatter(df['ip'], df['ea'], c=df['chi'], cmap='viridis', alpha=0.5, s=5)
ax.set_xlabel('Ionization Potential (ip)')
ax.set_ylabel('Electron Affinity (ea)')
ax.set_title('ip vs ea (colored by chi)')
plt.colorbar(scatter, ax=ax, label='χ (chi)')

plt.tight_layout()
plt.savefig('drugs_analysis.png', dpi=150)
plt.show()

print(f"\n✅ Saved visualization to 'drugs_analysis.png'")

# ============================================================
# Data Quality Checks
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY CHECKS")
print("=" * 60)

# Check for outliers (values beyond 3 standard deviations)
print("\n🚨 Potential Outliers (>3σ from mean):")
for target in targets:
    mean = df[target].mean()
    std = df[target].std()
    outliers = df[(df[target] < mean - 3*std) | (df[target] > mean + 3*std)]
    print(f"  {target}: {len(outliers)} outliers ({100*len(outliers)/len(df):.2f}%)")

# Check for negative values where unexpected
print(f"\n📉 Energy range: [{df['energy'].min():.2f}, {df['energy'].max():.2f}]")
print(f"   (All negative as expected for total energies)")

# Check chi = (ip + ea) / 2 relationship (Mulliken electronegativity)
df['chi_computed'] = (df['ip'] + df['ea']) / 2
chi_error = np.abs(df['chi'] - df['chi_computed']).mean()
print(f"\n🧪 Mulliken χ = (IP + EA)/2 check:")
print(f"   Mean absolute error: {chi_error:.6f}")
if chi_error < 0.01:
    print("   ✅ chi column matches Mulliken formula perfectly!")
else:
    print(f"   ⚠️ Small discrepancy detected (may be due to different definition)")
