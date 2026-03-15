# Experiment Tables

## Decision reliability
| Method | Acc (%) | FPR (%) | FNR (%) | Abstain (%) |
| --- | --- | --- | --- | --- |
| vanilla | 0.00 | 100.00 | 0.00 | 0.00 |
| text-rag | 0.00 | 100.00 | 0.00 | 0.00 |
| kg-rag | 0.00 | 100.00 | 0.00 | 0.00 |
| policy-kg-guardrails | 100.00 | 0.00 | 0.00 | 0.00 |

## Grounding and faithfulness
| Method | GF (%) | Rule Prec (%) | Attr Prec (%) | KGHallu (%) |
| --- | --- | --- | --- | --- |
| vanilla | 0.00 | 100.00 | 100.00 | 100.00 |
| text-rag | 0.00 | 100.00 | 100.00 | 100.00 |
| kg-rag | 0.00 | 100.00 | 100.00 | 100.00 |
| policy-kg-guardrails | 0.00 | 100.00 | 100.00 | 100.00 |

## Citation quality
| Method | CiteCorr (%) | CiteFaith (%) | Unsupported Claims (%) |
| --- | --- | --- | --- |
| vanilla | 100.00 | 0.00 | 100.00 |
| text-rag | 100.00 | 0.00 | 100.00 |
| kg-rag | 100.00 | 0.00 | 100.00 |
| policy-kg-guardrails | 100.00 | 0.00 | 100.00 |

## Adversarial robustness
| Method | Acc (%) | FPR (%) | GF (%) | Abstain (%) |
| --- | --- | --- | --- | --- |
| vanilla | 0.00 | 100.00 | 0.00 | 0.00 |
| text-rag | 0.00 | 100.00 | 0.00 | 0.00 |
| kg-rag | 0.00 | 100.00 | 0.00 | 0.00 |
| policy-kg-guardrails | 100.00 | 0.00 | 0.00 | 0.00 |

## Counterfactual quality
| Method | CF Validity (%) | CF Minimality (%) | Avg Attr Changes |
| --- | --- | --- | --- |
| vanilla | 0.00 | 0.00 | 0.67 |
| text-rag | 0.00 | 0.00 | 0.67 |
| kg-rag | 0.00 | 0.00 | 0.67 |
| policy-kg-guardrails | 0.00 | 0.00 | 0.67 |
