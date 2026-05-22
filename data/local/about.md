# About this RAG assistant

This is a study assistant for the Classic ML cycle (weeks 7-10) of the
mentoring program. The corpus contains the official documentation of the
scikit-learn library for every topic covered in the course.

## What topics I can answer questions about

I have indexed knowledge of the following scikit-learn modules:

- **Linear models**: ordinary least squares, Ridge, Lasso, ElasticNet,
  Bayesian Ridge, ARD, logistic regression (binary + multinomial),
  SGD, perceptron, RANSAC, Huber, quantile regression, polynomial features.
- **Decision trees**: classification trees, regression trees, multi-output,
  ID3 / C4.5 / CART criteria, gini and entropy, missing-value support,
  minimal cost-complexity pruning, practical tuning tips.
- **Ensembles**: gradient boosting (classic + histogram-based), random forests,
  extra trees, bagging meta-estimator, voting classifier, stacking,
  feature importance, monotonic constraints, learning-rate shrinkage.
- **Cross-validation**: KFold, repeated KFold, leave-one-out, leave-P-out,
  shuffle-split, stratified KFold, group KFold, time-series split,
  permutation test, picking iterators for imbalanced or grouped data.
- **Metrics**: confusion matrix, accuracy, precision, recall, F1, F-beta,
  ROC curve and AUC, PR curve and AUC, log-loss, MSE, MAE, R²,
  multiclass averaging (macro / micro / weighted), class-imbalance metrics.
- **Preprocessing**: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
  one-hot encoding, ordinal encoding, target encoding, normalisation,
  polynomial features, discretisation, power transforms.
- **Pipelines and composition**: Pipeline, ColumnTransformer, FeatureUnion,
  make_pipeline, make_column_transformer, set_output for pandas-out.
- **Hyperparameter tuning**: GridSearchCV, RandomizedSearchCV,
  HalvingGridSearchCV, scoring strategies, refit and best_estimator_.
- **Missing values**: SimpleImputer, IterativeImputer, KNNImputer,
  MissingIndicator, marker policies.
- **Feature selection**: SelectKBest, SelectFromModel, RFE, RFECV,
  variance threshold, mutual-information and chi-squared scoring.

## How to ask me questions

You can ask:

- Concept questions: *"What is the difference between Ridge and Lasso?"*,
  *"How does StratifiedKFold preserve class balance?"*
- Parameter questions: *"What does min_samples_leaf do?"*,
  *"What scoring options does GridSearchCV accept?"*
- How-to questions: *"How to score on multiple metrics in one cross_validate call?"*,
  *"How to build a Pipeline that scales numeric features and one-hot-encodes categorical ones?"*
- Comparison questions: *"When should I use HistGradientBoosting instead of GradientBoosting?"*,
  *"When is RandomizedSearchCV better than GridSearchCV?"*
- Production-tip questions: *"How do I prevent target leakage when doing cross-validation with preprocessing?"*

I respond in the same language as your question, so you can ask in
English or in Russian.

## What I CANNOT answer

The corpus does not cover deep learning, transformers, embeddings, RAG,
agents, time-series forecasting libraries (statsmodels / sktime),
recommendation systems, computer vision, or any other ML library besides
scikit-learn. If you ask about those — I will honestly say "I don't know".

This is by design: this assistant is the Classic ML study companion,
not a general ML chatbot.
