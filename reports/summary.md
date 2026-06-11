# Summary of Analysis so far


## EDA

* Completed EDA analysis to ensure I extract the right data for sentiment analysis. Further info:
    * [notebooks/exploratory_data_analysis.ipynb](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/notebooks/exploratory_data_analysis.ipynb)
    * [src/clean_data.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/clean_data.py)


## Sentiment Analysis

* Conducted sentiment analysis across time using a sample of lyrics from each year. I used BERT and gpt-40-mini to evaluate sentiment of the complete song. The analysis suggest lyrics have become more negative. It also may suggest that older songs in the dataset (1950s) may be bias towards more positive sounding tracks. For further info:
    * [reports/average_positive_sentiment_over_time_bert_trbs.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_positive_sentiment_over_time_bert_trbs.png)
    * [reports/average_sentiment_over_time_gpt_4o_mini.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_sentiment_over_time_gpt_4o_mini.png)
    * [src/sentiment_per_year_gpt_4o.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/sentiment_per_year_gpt_4o.py)
    * [src/sentiment_per_year_trbs_model.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/sentiment_per_year_trbs_model.py)
* Conducted sentiment analysis of hooks / common lines from lyrics. Used gpt-4o-mini to extract the top three most frequent lines from a song. A line is defined as sequence of more han two words on a new line. The analysis suggest lyrics have become more negative. For further info:
    * [reports/average_phrase_sentiment_over_time_gpt_4o_mini.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/average_phrase_sentiment_over_time_gpt_4o_mini.png)
    * [src/chorus_sentiment_per_year_gpt_4o.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/chorus_sentiment_per_year_gpt_4o.py)


## AI Safety & Bias

### Initial Genre Classification (eval_genre_bias.py)
* Explored how effectively LLMs classify genre based on music lyrics. Claude Sonnet 4 performed better than GPT-4o. Both struggled to classify pop and R&B — likely because the genre labels in this dataset are too high-level. For further info:
    * [src/eval_genre_bias.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/eval_genre_bias.py)
    * [reports/gpt_4o_genre_classification_confusion_matrix.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/gpt_4o_genre_classification_confusion_matrix.png)
    * [reports/confusion_matrix_genre_class_claude_sonnet_4.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/confusion_matrix_genre_class_claude_sonnet_4.png)

### Multi-Model Genre Classification (eval_genre_classification.py)
* Ran a systematic comparison of three Claude model tiers on genre classification: 50 songs per genre (250 total), equal samples across country, pop, r&b, rap, rock.

**Accuracy by model:**

| Model | Accuracy |
|---|---|
| Claude Haiku 4.5 | 65.6% |
| Claude Sonnet 4.6 | 69.9% |
| Claude Opus 4.6 | 73.6% |

Accuracy scales clearly with model tier — each step up adds ~4 percentage points.

**Per-genre F1 scores:**

| Genre | Haiku 4.5 | Sonnet 4.6 | Opus 4.6 |
|---|---|---|---|
| country | 0.61 | 0.83 | 0.84 |
| pop | 0.52 | 0.52 | 0.60 |
| r&b | 0.67 | 0.61 | 0.66 |
| rap | 0.80 | 0.82 | 0.84 |
| rock | 0.66 | 0.68 | 0.73 |

**Key findings:**
* **Rap is the most classifiable genre** across all models (recall 90–94%). Its lyrical style is likely the most distinctive.
* **Pop is the hardest genre** — consistently the lowest F1 across all models, mainly confused with rock and r&b. Dataset genre labels for pop may be too broad.
* **R&B is consistently confused with rap** — 12 of 50 r&b songs misclassified as rap by Haiku and Sonnet, 9 by Opus. Suggests meaningful lyrical overlap between these genres.
* **Country recall improves dramatically from Haiku (44%) to Sonnet/Opus (~76–80%)**, despite Haiku having near-perfect country precision (1.0). Haiku only predicts country when very confident, missing many true positives.
* The pop/R&B confusion observed in the initial evaluation persists across all model tiers, suggesting it is a property of the dataset labels rather than model capability.
* For further info:
    * [src/eval_genre_classification.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/eval_genre_classification.py)
    * [reports/confusion_matrix_claude-haiku-4-5-20251001_20260308_183628.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/confusion_matrix_claude-haiku-4-5-20251001_20260308_183628.png)
    * [reports/confusion_matrix_claude-sonnet-4-6_20260308_183628.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/confusion_matrix_claude-sonnet-4-6_20260308_183628.png)
    * [reports/confusion_matrix_claude-opus-4-6_20260308_183628.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/confusion_matrix_claude-opus-4-6_20260308_183628.png)


## Cultural Homogenisation Over Time

I am interested in whetehr increased use of LLMs is making our writing more homogenous, which “could diminish the collective diversity of creative ideas” (Moon et al 2025).

I decided to explore this idea by examining historical data to see if, prior to advanced AI systems, our writing is already becoming more homogenised. The idea here is to consider whether we can construct useful baselines for where society was pre-advanced AI (c.2022) and therefore estimate the impact of AI with more fidelity.

As a music enthusiast, I studied whether English language music lyrics have homogenised between 1950 and 2019.  I measured this by applying Kendal’s tau for monotonic trend testing over each half-decade. 

The Kaggle dataset is from the popular website genius.com, and contained rock, pop, R&B and country songs. I excluded rap from the analysis because the lyrical style is sufficiently different to warrant separate analysis.

I created two datasets based on user views: 75th percentile (825k songs) and 95th percentile (165k). This was to examine if homogenisation is related to song popularity. 

For each five-year bin, I sampled 2,000 songs and calculated the following metrics: 
-	Phrase repetition – using n-grams.
-	Vocabulary richness – using type-token-ratio (TTR) and moving average TTR with a fixed token length of 50 for fair comparison.
-	Thematic diversity – calculated pairwise cosine distance using: (1) TF-IDF and (2) sentence embeddings using the sentence transformer model all-MiniLM-L6-v2.

Using the stats package, I applied Kendall’s tau to each metric to measure monotonic trend. Kendall’s tau is appropriate here because the data is ranked by half-decade, is non-parametric and there are few data points (14). Under the null hypothesis, increases and decreases in a metric are equally likely.

The results showed that, for both datasets, thematic diversity monotonically decreased, for both TF-IDF and the sentence embeddings. Results for the other metrics were weakly significant or not significant. This suggests that thematic diversity has been narrowing for some time even before the advent of advanced AI. The results are shown below

**Headline result (75th percentile, 5-year bins, unweighted):**

| Metric | tau | p-value | Reading |
|---|---|---|---|
| Thematic diversity (TF-IDF) | −0.714 | **0.00014** | Strong, highly significant convergence |
| Thematic diversity (embeddings) | −0.692 | **0.00026** | Independently corroborates the convergence finding |
| TTR | −0.626 | 0.0012 | Apparent vocabulary narrowing (but see below) |
| MATTR (length-normalised) | −0.165 | 0.45 | **Null** — no narrowing once length is controlled |
| Fivegram repetition | +0.451 | 0.026 | Rising long-phrase repetition |
| Fourgram repetition | +0.363 | 0.079 | Borderline |
| Trigram repetition | +0.253 | 0.23 | Null |

There are several limitations of this study. Firstly, Kendall’s tau only measures trend not magnitude. Secondly, it is not clear what has narrowed. For example, is it language, themes (e.g. love, war) or something else. Thirdly, we need to research which events (technological or otherwise) may have contributed to homogenisation or not. 

Bibliography:
-	[20205, Moon et al. Homogenizing effect of large language models (LLMs) on creative diversity: An empirical comparison of human and ChatGPT writing](https://doi.org/10.1016/j.chbah.2025.100207)


**For further info:**
* [src/homogenisation_analysis.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/homogenisation_analysis.py)
* [reports/thematic_diversity_by_decade_75th.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/thematic_diversity_by_decade_75th.png)
* [reports/thematic_diversity_by_decade_95th.png](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/reports/thematic_diversity_by_decade_95th.png)


## AI Control Research — Multi-Model Fact Checking (PoC)

A proof-of-concept pipeline testing whether a verifier model (Claude Sonnet) can catch factual errors in a capable analyzer model's (Claude Opus) historical claims about song lyrics.

**Setup:**
* 30 songs from 1968–1972: 20 selected via protest keywords (war, peace, vietnam, freedom, etc.), 10 random popular songs by views
* Claude Opus (`claude-opus-4-6`) analyzed each song and extracted 3–5 specific, verifiable historical claims per song (150 claims total)
* Claude Sonnet (`claude-sonnet-4-6`) independently verified each claim: VERIFIED / CONTRADICTED / UNCERTAIN

**Overall Results (150 claims across 30 songs):**

| Verdict | Count | % |
|---|---|---|
| Verified | 104 | 69.3% |
| Contradicted | 38 | 25.3% |
| Uncertain | 8 | 5.3% |

**Protest vs. Random Popular Songs:**

| Song type | Songs | Claims | Verified | Contradicted | Uncertain |
|---|---|---|---|---|---|
| Protest keyword | 20 | 100 | 70% | 24% | 6% |
| Random popular | 10 | 50 | 68% | 28% | 4% |

Verification rates were nearly identical between song types, suggesting Opus's error rate is consistent regardless of whether it is analysing protest-era or mainstream material.

**Human Review Results (21 claims sampled — balanced across verdict types):**

| Model Verdict | Reviewed | Human: Correct | Human: Incorrect | Agreement Rate |
|---|---|---|---|---|
| Verified | 6 | 6 | 0 | 100% |
| Uncertain | 7 | 4 | 3 | 57% |
| Contradicted | 8 | 2 | 6 | 25% |
| **Overall** | **21** | **12** | **9** | **57%** |

**Key Findings:**
* The verifier model (Sonnet) was **most reliable on VERIFIED claims** — all 6 reviewed were confirmed correct by the human reviewer
* The verifier model was **least reliable on CONTRADICTED claims** — 6 out of 8 CONTRADICTED verdicts were assessed as incorrect by the human reviewer, suggesting the verifier over-contradicts (high false positive rate for flagging errors)
* **UNCERTAIN claims** were middling — the model was sometimes right to be cautious, but in several cases it flagged uncertainty where the claim was verifiable
* A recurring failure mode was **self-correction within the explanation**: the model's stated verdict did not match the conclusion reached in its own reasoning (seen in multiple Take It Easy, Evil Woman, and Leonard Cohen cases)
* When the verifier got CONTRADICTED wrong, it typically identified the right *topic* but cited an incorrect supporting detail in its reasoning — suggesting it can detect potentially wrong claims but is unreliable at pinning down exactly what is wrong
* Overall agreement of 57% indicates the verifier pipeline has **meaningful but limited reliability** as a standalone oversight mechanism at this sample size

**Implications for AI Control:**
* A single-model verifier is insufficient for high-stakes fact-checking — the high false positive rate on CONTRADICTED verdicts would generate too many spurious flags
* The VERIFIED verdict is the most trustworthy signal from this pipeline
* A more robust pipeline would require either a higher-confidence threshold before flagging CONTRADICTED, or a third model to adjudicate disputed claims

**For further info:**
* [src/ai_control_fact_check.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/ai_control_fact_check.py)
* [src/interactive_review.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/interactive_review.py)
* [src/analyze_agreement.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/analyze_agreement.py)
* [data/processed/ai_control_fact_check_results_reviewed.json](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/data/processed/ai_control_fact_check_results_reviewed.json)
