# About this RAG assistant

This is a study assistant for the Classic ML cycle (weeks 7-10) of the
mentoring program. The corpus contains the official documentation of
scikit-learn for three core topics covered in the course.

## What topics I can answer questions about

I have indexed knowledge of the following scikit-learn modules:

- **Linear models**: ordinary least squares, Ridge, Lasso, ElasticNet,
  logistic regression (binary + multinomial), SGD, regularisation
  strategies, the difference between L1 and L2 penalties, when to use
  which estimator, polynomial features.
- **Decision trees**: classification trees, regression trees,
  gini and entropy criteria, missing-value support, minimal
  cost-complexity pruning, practical tuning tips for `max_depth`,
  `min_samples_leaf`, `min_samples_split`.
- **Metrics**: confusion matrix, accuracy, precision, recall, F1, F-beta,
  ROC curve and AUC, PR curve and AUC, log-loss, MSE, MAE, R²,
  multiclass averaging (macro / micro / weighted), choosing a scoring
  function for imbalanced data.

## How to ask me questions

You can ask:

- Concept questions: *"What is the difference between Ridge and Lasso?"*,
  *"When does a decision tree overfit?"*
- Parameter questions: *"What does `min_samples_leaf` do?"*,
  *"What `average` values does `f1_score` accept?"*
- Formula questions: *"Show me the formula for Ridge regression"*,
  *"What is the precision/recall definition?"*
- How-to questions: *"How do I pick a scoring strategy for multiclass?"*,
  *"How do I prune a decision tree?"*
- Comparison questions: *"When is recall more important than precision?"*

I respond in the same language as your question, so you can ask in
English or in Russian.

## What I CANNOT answer (in this default build)

The default corpus does NOT cover ensembles, cross-validation,
preprocessing, pipelines, hyperparameter search, imputers, or feature
selection. Adding those modules is an optional homework — see the
"Расширение корпуса" section in your course materials.

Out of scope entirely (no matter how the corpus is extended): deep
learning, transformers, embeddings, RAG, agents, time-series libraries
like statsmodels / sktime, recommendation systems, computer vision,
and any other ML library besides scikit-learn. If you ask about those
— I will honestly say "I don't know".

This is by design: this assistant is the Classic ML study companion,
not a general ML chatbot.
