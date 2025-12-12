# UCU: Ukrainian emotions competition

For this classification task, I have started with simple TF-IDF vectorizer + logistic regression on a train/val split from the train data. I was able to get following results;
- Validation F1 - Emotion: 0.3274
- Validation F1 - Category: 0.4864
- Overall F1 (competition metric): 0.4069
- Kaggle test submission: 0.4220

My next attempt was multilingual BERT from HuggingFace, which I trained for 3 epochs on train/val split:
- Validation F1 - Emotion: 0.3840
- Validation F1 - Category: 0.7157
- Overall F1: 0.5498
- Kaggle test submission: 0.5581
Before moving to full dataset training or increasing epochs, I wanted to try a few more models.

Ukr-roBERTa-base did not seem to do better than previous BERT, despite the fact that I ran it for a few more epochs:
- Validation F1 - Emotion: 0.5189
- Validation F1 - Category: 0.7536
- Overall F1: 0.6363
- Kaggle test submission: 0.5660

So I also added XLM-roBERTa-Large and tried equally weighted ensamble of these three models, which finally got me to 0.6+ on Kaggle.

For the final round of training on a full dataset, I also added different weight combinations:

```
weight_combinations = [
    (0.33, 0.33, 0.34),  # Equal weights
    (0.2, 0.3, 0.5),     # More weight to XLM
    (0.25, 0.25, 0.5),   # XLM dominant
    (0.3, 0.3, 0.4),     # Balanced with slightly higher XML
    (0.2, 0.4, 0.4),     # UKR + XLM
    (0.15, 0.35, 0.5),   # XLM very dominant
    (0.25, 0.35, 0.4),   # Dominant XLM with slightly increased BERT
    (0.3, 0.2, 0.5),     # BERT + XLM
]
```

And got the following results:
- Weights 0.33, 0.33, 0.34 -> F1: 0.9604
- Weights 0.20, 0.30, 0.50 -> F1: 0.9690
- Weights 0.25, 0.25, 0.50 -> F1: 0.9685
- Weights 0.30, 0.30, 0.40 -> F1: 0.9618
- Weights 0.20, 0.40, 0.40 -> F1: 0.9631
- Weights 0.15, 0.35, 0.50 -> F1: 0.9696
- Weights 0.25, 0.35, 0.40 -> F1: 0.9634
- Weights 0.30, 0.20, 0.50 -> F1: 0.9687

Overall weights did not matter as much, and this got me my best 0.7 Kaggle test submission.