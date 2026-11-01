# Customer Churn Model Evaluation Report

- Dataset rows: **7,043**
- Input columns before engineering: **19**
- Best model: **Logistic Regression**
- Selection: **mean cross-validation PR-AUC**
- Decision threshold: **0.570**

## Model comparison

| Model | CV PR-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6638 | 0.7339 | 0.4992 | 0.7914 | 0.6122 | 0.8425 | 0.6371 |
| Gradient Boosting | 0.6634 | 0.8006 | 0.6609 | 0.5107 | 0.5762 | 0.8458 | 0.6653 |
| Neural Network (MLP) | 0.6486 | 0.8020 | 0.6740 | 0.4920 | 0.5688 | 0.8407 | 0.6261 |
| Random Forest | 0.6412 | 0.7622 | 0.5402 | 0.7005 | 0.6100 | 0.8350 | 0.6368 |
| Extra Trees | 0.6196 | 0.7608 | 0.5407 | 0.6578 | 0.5935 | 0.8264 | 0.6260 |

## Best model held-out test metrics

```json
{
  "accuracy": 0.7679205110007097,
  "balanced_accuracy": 0.7583559378955798,
  "precision": 0.5465346534653466,
  "recall": 0.7379679144385026,
  "f1": 0.6279863481228669,
  "roc_auc": 0.8425146606732284,
  "pr_auc": 0.637119404332333,
  "mcc": 0.4758000845985793,
  "threshold": 0.5700000000000001,
  "confusion_matrix": [
    [
      806,
      229
    ],
    [
      98,
      276
    ]
  ],
  "classification_report": {
    "No Churn": {
      "precision": 0.8915929203539823,
      "recall": 0.778743961352657,
      "f1-score": 0.8313563692625064,
      "support": 1035.0
    },
    "Churn": {
      "precision": 0.5465346534653466,
      "recall": 0.7379679144385026,
      "f1-score": 0.6279863481228669,
      "support": 374.0
    },
    "accuracy": 0.7679205110007097,
    "macro avg": {
      "precision": 0.7190637869096644,
      "recall": 0.7583559378955798,
      "f1-score": 0.7296713586926866,
      "support": 1409.0
    },
    "weighted avg": {
      "precision": 0.800001868674529,
      "recall": 0.7679205110007097,
      "f1-score": 0.7773745467598626,
      "support": 1409.0
    }
  }
}
```

## Notes

The hold-out test set is used only for final evaluation. Model selection uses cross-validation on the training split. Metrics describe this dataset and split; they are not a guarantee of production performance.