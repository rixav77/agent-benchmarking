# Failure Analysis Report

## Failure Category Summary

| Category | Description | Simple RAG | Agentic RAG | Delta |
|----------|-------------|-----------|------------|-------|
| CAT A | False Positive (unanswerable answered) | 219 | 194 | -25 |
| CAT B | False Negative (answerable refused) | 63 | 61 | -2 |
| CAT C | Wrong Answer (all tiers rejected) | 27 | 25 | -2 |
| CAT D | Faithfulness Failure | 15 | 15 | +0 |


## Category A: False Positive (Unanswerable Answered)

- Simple RAG: 219 (21.9%)
- Agentic RAG: 194 (19.4%)

**Simple RAG examples:**

  - Q: "What meetings are closed to the public?"
    Gold: (unanswerable)
    Pred: Closed plenary sessions of the IPCC and certain meetings of the IPCC Bureau, wor
    F1=0.000, Tier=none
  - Q: "What are sometimes present in the plug's firebox crown?"
    Gold: (unanswerable)
    Pred: Lead fusible plugs may be present in the crown of the boiler's firebox.
    F1=0.000, Tier=none

**Agentic RAG examples:**

  - Q: "What meetings are closed to the public?"
    Gold: (unanswerable)
    Pred: Closed plenary sessions of the IPCC and meetings of the IPCC Bureau, workshops, 
    F1=0.000, Tier=none
  - Q: "What are sometimes present in the plug's firebox crown?"
    Gold: (unanswerable)
    Pred: Lead fusible plugs
    F1=0.000, Tier=none


## Category B: False Negative (Answerable Refused)

- Simple RAG: 63 (6.3%)
- Agentic RAG: 61 (6.1%)

**Simple RAG examples:**

  - Q: "How old were the fossils found in China?"
    Gold: 515 million years, about 515 million years
    Pred: unanswerable
    F1=0.000, Tier=none
  - Q: "Who was the opposing party in the Runyon case?"
    Gold: McCrary
    Pred: unanswerable
    F1=0.000, Tier=none

**Agentic RAG examples:**

  - Q: "In what calcium containing body part is oxygen a part?"
    Gold: bones
    Pred: unanswerable
    F1=0.000, Tier=none
  - Q: "How old were the fossils found in China?"
    Gold: 515 million years, about 515 million years
    Pred: unanswerable
    F1=0.000, Tier=none


## Category C: Wrong Answer (All Tiers Rejected)

- Simple RAG: 27 (2.7%)
- Agentic RAG: 25 (2.5%)

**Simple RAG examples:**

  - Q: "When would a person be considered to be excising a constitutional impasse?"
    Gold: the head of government would be acting in her or his capacit
    Pred: A person would not be excising a constitutional impasse, as constitutional impas
    F1=0.250, Tier=none
  - Q: "How many Americans are richer than more than half of all citizens?"
    Gold: 400
    Pred: The top 400 richest Americans have more wealth than half of all Americans combin
    F1=0.143, Tier=none

**Agentic RAG examples:**

  - Q: "How many Americans are richer than more than half of all citizens?"
    Gold: 400
    Pred: The top 400 richest Americans have more wealth than half of all Americans combin
    F1=0.143, Tier=none
  - Q: "Where is the Rhine Bridge?"
    Gold: Arnhem
    Pred: The Rhine Bridge is located between Emmerich and Cleves, and it is the longest s
    F1=0.000, Tier=none


## Category D: Faithfulness Failure

- Simple RAG: 15 (1.5%)
- Agentic RAG: 15 (1.5%)

**Simple RAG examples:**

  - Q: "Where is the Rhine Bridge?"
    Gold: Arnhem
    Pred: The Rhine Bridge is located between Emmerich and Cleves.
    F1=0.000, Tier=none
  - Q: "What made emigration to these colonies attractive?"
    Gold: they were accepted and allowed to worship freely, allowed to
    Pred: Land made available to migrants from Europe and the colonies further south.
    F1=0.211, Tier=none

**Agentic RAG examples:**

  - Q: "Where is the Rhine Bridge?"
    Gold: Arnhem
    Pred: The Rhine Bridge is located between Emmerich and Cleves, and it is the longest s
    F1=0.000, Tier=none
  - Q: "Where does the Lek join? "
    Gold: Noord River, Nieuwe Maas
    Pred: The Lek joins the North Sea through the former Meuse estuary, near Rotterdam.
    F1=0.000, Tier=none


## Unanswerable Detection Performance

| Metric | Simple RAG | Agentic RAG |
|--------|-----------|------------|
| Precision | 0.817 | 0.834 |
| Recall | 0.562 | 0.612 |
| F1 | 0.666 | 0.706 |

| | Simple RAG | Agentic RAG |
|--|-----------|------------|
| True Positive | 281 | 306 |
| False Positive | 63 | 61 |
| False Negative | 219 | 194 |
| True Negative | 437 | 439 |
