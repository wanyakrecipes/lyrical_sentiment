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

A study of whether popular music lyrics have homogenised since 1950, framed for an AI Safety audience: if popular music has measurably converged under the pre-AI engagement and recommendation regime, generative systems optimising the same engagement signal would by the same logic accelerate that convergence. Narrowing thematic diversity represents a reduction in the range of ideas and emotional registers people are exposed to — a societal-resilience concern.

**Setup:**
* Dataset: cleaned Genius lyrics, English only, 1950–2019
* Two view-count filters tested: 95th percentile (~165k songs) and 75th percentile (~5× larger)
* Two time-bin resolutions tested: decade and 5-year bins
* Stratified sampling of 2,000 songs per bin with `random_state=42`
* Rap excluded — its distinctive lyrical density would otherwise dominate the trend
* Seven metrics:
    * TTR (type-token ratio) and MATTR (window=50, length-normalised) — vocabulary richness
    * Trigram, fourgram, fivegram repetition rates — phrase repetition
    * Mean pairwise cosine distance on TF-IDF (5,000 features) within each bin — thematic diversity (vocabulary overlap)
    * Mean pairwise cosine distance on sentence embeddings (all-MiniLM-L6-v2, 384-dim) within each bin — thematic diversity (semantic similarity; recognises shared themes expressed with different vocabulary)
* Kendall's tau for monotonic trend testing
* View-count-weighted re-run using log(views + 1) to test for popularity amplification

The 75th-percentile / 5-year-bin combination (n = 14 bins) is the primary result — finer tau resolution and most bins above the 100-song low-confidence threshold.

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

**Key findings:**

* **Evideence for thematic convergence is replicated across two independent methods.** Pairwise cosine distance between songs within a time bin has declined monotonically since 1950 on *both* TF-IDF (vocabulary overlap; tau = −0.714, p = 0.00014) and sentence embeddings (semantic similarity; tau = −0.692, p = 0.00026). Because the embedding measure recognises shared themes even when expressed with different words, its agreement rules out the most obvious "it's just a vocabulary artefact" objection to the TF-IDF result. The trend is robust across both percentile filters: on the 95th-percentile dataset the embedding method is the single strongest signal in the entire study (tau = −0.868, p = 5.5×10⁻⁷), with TF-IDF close behind (tau = −0.692, p = 0.00026).
* **The secondary findings are weaker and heavily caveated.** The "vocabulary is narrowing" story does not survive length control: TTR declines (p = 0.0012) but MATTR, which normalises for length with a rolling 50-token window, shows essentially no trend (p = 0.45), so the apparent narrowing was a lyric-length artefact — songs got longer, not less varied per word. Phrase repetition narrows to a single robust dimension: at 5-year resolution only fivegram (hook-length) repetition rises significantly, while trigram and fourgram trends fall out of significance. And the 'algorithmic amplification' effect did not robustly replicate — all `tau_delta` values between unweighted and view-weighted runs are within ±0.07 (below the test's noise floor at n = 91 pairs), so the earlier decade-level signal was likely an artefact of Kendall's tau's coarse discrete grid at n = 7.
* **The 75th percentile dataset matters.** Expanding from the 95th to 75th percentile gave finer tau resolution, lifted older bins above the low-confidence threshold, and most importantly exposed the TTR/MATTR divergence that the curated 95th-percentile sample had masked.

**Implications for AI Safety:**

The single robust finding — thematic convergence — is the dimension that maps most directly onto the societal-resilience framing in [AISI's research agenda on frontier AI risks](https://www.aisi.gov.uk). Narrowing thematic diversity represents a reduction in the *range* of ideas and emotional registers people are exposed to via popular media, independent of whether individual songs are lexically richer or poorer. 

<!-- TODO: look at the research on gradual disempowerment and add some colour here.

Pre-AI recommendation, marketing, and engagement-optimisation infrastructure have already produced measurable convergence on this dimension. Generative systems optimising the same engagement signal at scale would, by the same logic, accelerate it — and theme is the easiest of the three measured dimensions for current generative models to influence at population scale.

The vocabulary and amplification claims from earlier iterations should be either dropped or heavily caveated. The hook-level repetition finding can stay but should be narrowed in scope. -->

**Limitations:**
* n = 14 bins is sufficient for trend testing but not for fine-grained sub-trend analysis
* Kendall's tau measures rank order, not magnitude — small `tau_delta` between runs does not mean per-bin numbers are identical
* Pre-2000s view counts are a noisy popularity proxy regardless of percentile filter
* This is correlational; "algorithmic amplification" here refers to engagement-optimisation logic broadly, not modern recommender systems specifically

**For further info:**
* [src/homogenisation_analysis.py](https://github.com/wanyakrecipes/lyrical_sentiment/blob/main/src/homogenisation_analysis.py)

<!-- TODO - add the 75th and 95th percentile to the `reports/` -->


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
