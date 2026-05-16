"""
UX Sentiment Analyzer
=====================
Sentiment analysis of mobile app reviews with a UX design perspective.
Author: Ivan Ilin
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import re

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────
C_POS  = "#4A90D9"
C_NEU  = "#F5A623"
C_NEG  = "#E85D5D"
C_DARK = "#1A1A2E"
PALETTE = [C_POS, C_NEU, C_NEG]
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv("reviews.csv")
print(f"Dataset: {len(df)} reviews · {df['app'].nunique()} apps\n")

# ══════════════════════════════════════════════════════════════════════
# 2. VADER SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
analyzer = SentimentIntensityAnalyzer()

def vader_label(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:  return "positive"
    elif score <= -0.05: return "negative"
    else: return "neutral"

def vader_score(text):
    return analyzer.polarity_scores(str(text))["compound"]

df["predicted_sentiment"] = df["review"].apply(vader_label)
df["compound_score"]      = df["review"].apply(vader_score)
print("Sentiment distribution (predicted):")
print(df["predicted_sentiment"].value_counts(), "\n")

# ══════════════════════════════════════════════════════════════════════
# 3. FIGURE 1 — Rating vs Sentiment compound score
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("App Reviews: Ratings & Sentiment Scores", fontsize=14,
             fontweight="bold", color=C_DARK, y=1.01)

# left: count by predicted sentiment
sent_counts = df["predicted_sentiment"].value_counts().reindex(
    ["positive", "neutral", "negative"])
bars = axes[0].bar(sent_counts.index, sent_counts.values,
                   color=PALETTE, edgecolor="white", linewidth=0.8, width=0.5)
axes[0].set_title("Predicted Sentiment Distribution", fontsize=11, color=C_DARK)
axes[0].set_ylabel("Number of Reviews")
axes[0].set_ylim(0, sent_counts.max() * 1.18)
for bar, val in zip(bars, sent_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 str(val), ha="center", fontsize=10, color=C_DARK, fontweight="bold")

# right: violin of compound score by star rating
palette_5 = {1: C_NEG, 2: "#F07A5A", 3: C_NEU, 4: "#72BF78", 5: C_POS}
parts = axes[1].violinplot(
    [df[df["rating"] == r]["compound_score"].values for r in range(1, 6)],
    positions=range(1, 6), showmedians=True, showmeans=False)
for i, (pc, r) in enumerate(zip(parts["bodies"], range(1, 6))):
    pc.set_facecolor(palette_5[r]); pc.set_alpha(0.75)
parts["cmedians"].set_color(C_DARK); parts["cmedians"].set_linewidth(1.5)
axes[1].set_title("VADER Score by Star Rating", fontsize=11, color=C_DARK)
axes[1].set_xlabel("Star Rating"); axes[1].set_ylabel("Compound Score (−1 … +1)")
axes[1].set_xticks(range(1, 6))
axes[1].axhline(0, color="#aaa", linewidth=0.8, linestyle="--")

plt.tight_layout()
plt.savefig("results/fig1_rating_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1")

# ══════════════════════════════════════════════════════════════════════
# 4. FIGURE 2 — Per-app sentiment breakdown
# ══════════════════════════════════════════════════════════════════════
app_sent = (df.groupby(["app", "predicted_sentiment"])
              .size().unstack(fill_value=0)
              .reindex(columns=["positive", "neutral", "negative"], fill_value=0))
app_sent_pct = app_sent.div(app_sent.sum(axis=1), axis=0) * 100
app_sent_pct = app_sent_pct.sort_values("positive", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
app_sent_pct[["negative", "neutral", "positive"]].plot(
    kind="barh", stacked=True, ax=ax,
    color=[C_NEG, C_NEU, C_POS], edgecolor="white", linewidth=0.5)
ax.set_title("Sentiment Breakdown by App (%)", fontsize=13,
             fontweight="bold", color=C_DARK)
ax.set_xlabel("Share of Reviews (%)")
ax.set_ylabel("")
ax.legend(["Negative", "Neutral", "Positive"], loc="lower right", framealpha=0.9)
ax.axvline(50, color="#aaa", linewidth=0.8, linestyle="--")
for i, (idx, row) in enumerate(app_sent_pct.iterrows()):
    pos_pct = row["positive"]
    ax.text(101, i, f"{pos_pct:.0f}% 👍", va="center", fontsize=8.5, color=C_DARK)
plt.tight_layout()
plt.savefig("results/fig2_app_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")

# ══════════════════════════════════════════════════════════════════════
# 5. FIGURE 3 — UX keyword frequency in negative reviews
# ══════════════════════════════════════════════════════════════════════
UX_KEYWORDS = {
    "navigation": ["navigation", "navigate", "menu", "find", "buried"],
    "performance": ["crash", "slow", "freeze", "bug", "loading", "broken"],
    "ads":         ["ads", "ad", "advertisement", "popup"],
    "design":      ["ui", "design", "layout", "ugly", "outdated", "cluttered"],
    "onboarding":  ["onboarding", "confusing", "complicated", "steps", "setup"],
    "notifications":["notification", "notify", "alert", "configure"],
}

neg_reviews = df[df["predicted_sentiment"] == "negative"]["review"].str.lower()
kw_counts = {}
for category, words in UX_KEYWORDS.items():
    count = sum(neg_reviews.str.contains("|".join(words), regex=True))
    kw_counts[category] = count

fig, ax = plt.subplots(figsize=(9, 5))
cats   = list(kw_counts.keys())
counts = list(kw_counts.values())
colors_bar = [C_NEG if c == max(counts) else "#AABCF5" for c in counts]
bars = ax.bar(cats, counts, color=colors_bar, edgecolor="white", linewidth=0.8, width=0.5)
ax.set_title("Top UX Pain Points in Negative Reviews", fontsize=13,
             fontweight="bold", color=C_DARK)
ax.set_ylabel("Number of Mentions")
ax.set_ylim(0, max(counts) * 1.2)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha="center", fontsize=10, fontweight="bold", color=C_DARK)
plt.tight_layout()
plt.savefig("results/fig3_ux_pain_points.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3")

# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE 4 — Confusion matrix (VADER vs. true labels)
# ══════════════════════════════════════════════════════════════════════
labels = ["positive", "neutral", "negative"]
cm = confusion_matrix(df["true_sentiment"], df["predicted_sentiment"], labels=labels)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="#ddd", ax=ax,
            annot_kws={"fontsize": 11})
ax.set_title("Confusion Matrix: VADER Predictions (%)", fontsize=12,
             fontweight="bold", color=C_DARK)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("results/fig4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4")

# ══════════════════════════════════════════════════════════════════════
# 7. CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════
report = classification_report(
    df["true_sentiment"], df["predicted_sentiment"],
    labels=labels, digits=3)
print("\nClassification Report:\n", report)

# Save summary stats
summary = {
    "total_reviews":    int(len(df)),
    "apps_analyzed":    int(df["app"].nunique()),
    "accuracy":         float((df["true_sentiment"] == df["predicted_sentiment"]).mean()),
    "avg_compound":     float(df["compound_score"].mean()),
    "top_pain_point":   max(kw_counts, key=kw_counts.get),
    "most_positive_app": app_sent_pct["positive"].idxmax(),
    "most_negative_app": app_sent_pct["positive"].idxmin(),
}
import json
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ All outputs saved to results/")
print(json.dumps(summary, indent=2))
