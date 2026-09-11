# Loan Default Risk Model

A logistic regression model that estimates the probability a loan applicant will default, built as a decision-support tool for underwriters rather than an automated approve/deny system. Developed as a business case study for a fictional community bank ("FirstState") facing rising default rates.

## Business problem

FirstState Community Bank's default rate had risen above 10%, up from a historical 6-7%, with each 1% increase costing an estimated $1.4M annually. The bank needed a way to flag high-risk applications for senior review without rejecting good borrowers or damaging its relationship-based lending model.

## Data

2,000 loan applications with 6 features (`annual_income`, `credit_score`, `debt_to_income_ratio`, `loan_amount`, `employment_years`, `has_prior_default`) and a binary target (`defaulted`). The target is imbalanced - only 9.2% of loans in the dataset defaulted.

## Approach

- **Model**: Logistic regression, chosen for interpretability - it outputs odds ratios that explain why an applicant is flagged as risky, which matters for underwriters who need to justify decisions.
- **Split**: 70/30 train/test, stratified on the target to preserve the 9.2% default rate in both sets.
- **Metric selection**: Accuracy is misleading here - a model that predicts "repaid" for everyone would already be ~91% accurate while catching zero defaults. Recall on the default class was prioritized instead, since a missed default (~$18,500 average loss) costs roughly 13x more than a false alarm (~$1,400 in lost interest and review time).
- **Threshold tuning**: The default 0.50 classification threshold caught only 1 of 55 defaults in the test set (1.8% recall) - far too conservative for this cost structure. Lowering the threshold to 0.20 raised recall to 40% (22/55 caught), reducing estimated test-set error cost from ~$1,000,400 to ~$683,300.

## Key findings

- Debt-to-income ratio and prior default history were the strongest predictors of default risk; credit score and annual income were the strongest protective factors.
- The bank's own cost asymmetry (missed default much greater than false alarm) justified deliberately trading precision for recall - the "right" threshold depends on business costs, not a default 0.50 cutoff.
- Recommended deployment: a three-tier risk workflow (low / moderate / high risk) routing applications to different levels of human review, with no automatic denials - the model acts as a screening aid, not a decision-maker.

## Limitations

The dataset is a static snapshot with no macroeconomic context, no loan-purpose or collateral information, and no time-series signal (e.g. worsening cash flow trends). See the notebook's Limitations & Next Steps section for the full discussion, including a proposed "shadow mode" deployment to validate the model against real outcomes before live use.

## Stack

Python, pandas, scikit-learn (`LogisticRegression`, `train_test_split`, classification metrics), matplotlib, seaborn.

## Notebook

[`loan_default_risk_analysis.ipynb`](loan_default_risk_analysis.ipynb) - full analysis, from data exploration through model evaluation and business recommendations.
