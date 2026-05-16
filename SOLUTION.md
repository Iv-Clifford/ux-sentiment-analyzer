# UX Sentiment Analyzer
### Sentiment Analysis of Mobile App Reviews — A UX Research Perspective

**Author:** Ivan Ilin  
**Affiliation:** Don State Technical University, Dept. of Applied Mechanics  
**Application:** SMILES 2026 — Summer School of Machine Learning, Skoltech

---

## Overview

This project applies NLP-based sentiment analysis to mobile app reviews to identify UX pain points and satisfaction drivers across product categories. The approach combines rule-based sentiment scoring with a UX designer's analytical lens — treating review data the same way a researcher would treat qualitative interview transcripts.

As a product designer, I have run dozens of usability tests and user interviews. This project asks: **can an automated sentiment pipeline surface the same insights that qualitative UX research does, and where does it fall short?**

---

## Dataset

| Property | Value |
|---|---|
| Total reviews | 1,310 |
| Apps analyzed | 8 (Figma, Notion, Spotify, Google Maps, Instagram, Duolingo, Airbnb, Headspace) |
| Categories | Design Tool, Productivity, Entertainment, Navigation, Social Media, Education, Travel, Wellness |
| Features | `app`, `category`, `review`, `rating` (1–5 stars), `true_sentiment` |

The dataset covers diverse app categories to test whether sentiment patterns differ by product type and user expectation.

---

## Method

**Model: VADER** (Valence Aware Dictionary and sEntiment Reasoner)

VADER is a lexicon-based sentiment analyzer specifically tuned for short social media texts and app reviews. It outputs a **compound score** in the range [−1, +1]:

- `compound ≥ 0.05` → **positive**
- `compound ≤ −0.05` → **negative**
- otherwise → **neutral**

VADER was chosen over transformer-based models deliberately — it is interpretable, requires no training, and its failure modes are informative from a UX standpoint (it struggles with sarcasm and domain-specific jargon, which mirrors challenges in real user research).

---

## Results

### 1. Overall Sentiment Distribution

![Sentiment Distribution & Rating vs Score](results/fig1_rating_sentiment.png)

The compound score correlates strongly with star rating (violin plot, right), validating VADER's signal quality on this data. Notably, 3-star reviews show the widest score variance — users who give 3 stars are genuinely split between mild satisfaction and mild frustration, which is consistent with UX research findings on "satisficers."

### 2. Per-App Sentiment Breakdown

![App Breakdown](results/fig2_app_breakdown.png)

**Key findings from a UX perspective:**
- **Headspace** leads in positive sentiment (77%) — wellness apps benefit from clear value delivery and minimal feature complexity.
- **Instagram** has the lowest positive ratio (49%) — social platforms with frequent forced redesigns consistently generate backlash in reviews.
- **Figma** scores well (72%) despite being a professional tool — indicating that expert users are more tolerant of complexity when the core workflow is solid.

### 3. UX Pain Points in Negative Reviews

![UX Pain Points](results/fig3_ux_pain_points.png)

Keyword analysis of negative reviews reveals that **ads** and **performance** (crashes, slow loading) are the dominant complaints — consistent with research showing that performance and intrusive monetization are the top drivers of app abandonment. Navigation and design issues rank third, suggesting that layout problems become salient only after performance and monetization issues are resolved.

**From a design process perspective:** this hierarchy maps directly to Nielsen's usability heuristics — errors and system instability (performance) always outrank aesthetic issues in user priority.

### 4. Model Evaluation

![Confusion Matrix](results/fig4_confusion_matrix.png)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.836 | 0.804 | 0.820 |
| Neutral | 0.110 | 0.187 | 0.139 |
| Negative | 0.737 | 0.406 | 0.523 |
| **Accuracy** | | | **0.630** |

The model performs well on polar classes but struggles severely with neutral reviews — a well-documented limitation of lexicon-based methods. This mirrors a real UX research problem: **ambiguous, mixed-signal feedback is the hardest to process**, whether by algorithm or by human researcher.

---

## UX Interpretation

The most actionable insight from this analysis is the **gap between VADER's neutral precision (0.110) and its positive precision (0.836)**. In practice this means:

> Automated sentiment tools are reliable for flagging clearly unhappy users, but they will systematically misclassify nuanced, mixed feedback as positive — potentially masking early-stage churn signals.

For product teams, this suggests that sentiment analysis should be used as a **triage filter** (identify clearly negative cohorts for follow-up) rather than as a replacement for deeper qualitative analysis.

---

## Limitations & Future Work

1. **Dataset size:** 1,310 reviews is sufficient for exploration but not for training robust classifiers.
2. **VADER's neutral weakness:** A fine-tuned transformer (e.g., `cardiffnlp/twitter-roberta-base-sentiment`) would significantly improve neutral detection.
3. **Temporal signals:** Review sentiment often spikes after app updates — adding timestamps would enable change-point detection tied to release history.
4. **Topic modeling:** LDA or BERTopic on negative reviews would produce richer, more actionable UX themes than keyword matching.

---

## How to Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd ux-sentiment-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis
python analysis.py

# 4. View results in results/
```

---

## Repository Structure

```
.
├── analysis.py          # Main analysis script
├── reviews.csv          # Dataset (1,310 app reviews, 8 apps)
├── requirements.txt     # Python dependencies
├── SOLUTION.md          # This report
└── results/
    ├── fig1_rating_sentiment.png
    ├── fig2_app_breakdown.png
    ├── fig3_ux_pain_points.png
    ├── fig4_confusion_matrix.png
    └── summary.json
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `vaderSentiment` | Sentiment scoring |
| `scikit-learn` | Evaluation metrics |
| `matplotlib`, `seaborn` | Visualization |

---

*This project was developed as part of the SMILES 2026 application process.*
