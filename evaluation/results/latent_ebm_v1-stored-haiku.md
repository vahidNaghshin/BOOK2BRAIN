# Evaluation of stored extractions: latent_ebm_v1

## latent_ebm_v1 v1.0.1 — anthropic.claude-haiku-5.5, stormy-claude-haiku-5.5

Reference status: **draft** · dataset fingerprint `3421a1336c2d0f1d` · passages extracted: 5 of 6

| Run | Status | Configuration | Provider / model | Prompt versions | Parameters | Duration | Tokens in / out | Est. cost |
|---|---|---|---|---|---|---|---|---|
| `7ab98233` | completed | n/a | n/a / anthropic.claude-haiku-5.5 | concept_extraction@1.1.0, equation_extraction@1.1.0 | `{"categories": ["concept", "equation"], "context_passages": 1, "generation": {"max_output_tokens": 4096, "structured_output": "none", "temperature": 0.0}, "limit": 2, "offset": 10}` | 44.1 s | n/a / n/a | n/a |
| `85a15ec8` | needs_review | stormy-claude-haiku-5.5 | stormy / anthropic.claude-haiku-5.5 | concept_extraction@1.2.0, principle_extraction@1.2.0 | `{"categories": ["concept", "principle"], "context_passages": 1, "generation": {"max_output_tokens": 4096, "structured_output": "none", "temperature": 0.0}, "limit": 2, "offset": 0}` | 64.1 s | 35255 / 11071 | n/a |
| `703ae243` | completed | stormy-claude-haiku-5.5 | stormy / anthropic.claude-haiku-5.5 | concept_extraction@1.2.0 | `{"categories": ["concept"], "context_passages": 1, "generation": {"max_output_tokens": 4096, "structured_output": "none", "temperature": 0.0}, "limit": 1, "offset": 0}` | 6.3 s | 4106 / 990 | n/a |
| `35e84308` | needs_review | stormy-claude-haiku-5.5 | stormy / anthropic.claude-haiku-5.5 | concept_extraction@1.2.0, equation_extraction@1.2.0, principle_extraction@1.2.0, relationship_extraction@1.0.0 | `{"categories": ["concept", "principle", "equation"], "context_passages": 1, "extract_relationships": true, "generation": {"max_output_tokens": 4096, "structured_output": "none", "temperature": 0.0}, "limit": null, "offset": 0}` | 82.4 s | 47374 / 13378 | n/a |

### A. Completeness

| Metric | Result | Note |
|---|---|---|
| Required units found (recall) | 88.2% (15/17) |  |
| Recall: concept | 100.0% (9/9) |  |
| Recall: method | 100.0% (1/1) |  |
| Recall: principle | 71.4% (5/7) |  |
| Recall in passages with several required units | 85.7% (12/14) |  |
| Extracted units explained by the reference (precision) | 74.1% (20/27) | 1 unmatched unit(s) in non-exhaustive passages are not counted either way: a reviewer must judge them. |
| Passages without knowledge left without unexpected units | 0.0% (0/1) |  |
| Required units that seem merged into another unit | 1 | Candidates for a reviewer: similar but distinct units may have been merged. |

### B. Semantic accuracy

| Metric | Result | Note |
|---|---|---|
| Key facts present in matched units (lexical proxy) | 100.0% (60/60) | A reviewer must still read the matched pairs; this does not show that meaning was preserved. |
| Matched units containing a phrase the reviewer ruled out | 0 |  |
| Units the pipeline flagged as unsupported | 3 |  |
| Required relationships found | 33.3% (1/3) |  |
| Found relationships of an accepted type | 100.0% (1/1) |  |
| Relationships between recognised units that the reference expects | 50.0% (1/2) |  |

### C. Source accuracy

| Metric | Result | Note |
|---|---|---|
| Matched units citing the right passage | 100.0% (20/20) |  |
| Matched units with pages inside the passage's pages | 100.0% (20/20) |  |
| Units whose sources verified | 100.0% (35/35) |  |
| Units whose primary source is the passage that was extracted | 71.4% (25/35) |  |
| Quotes found in their passage | 100.0% (37/37) |  |

### D. Structural validity

| Metric | Result | Note |
|---|---|---|
| Answers that matched their schema | 71.4% (15/21) |  |
| Extractions that succeeded on the first call | 64.7% (11/17) |  |
| Matched units in the expected category | 85.0% (17/20) |  |

### E. Duplication

| Metric | Result | Note |
|---|---|---|
| Redundant units among stored units | 17.1% (6/35) |  |
| Duplicates the pipeline refused to store | 10 |  |

### F. Equations

| Metric | Result | Note |
|---|---|---|
| Reference equations found | 100.0% (5/5) |  |
| Original expression preserved | 100.0% (5/5) |  |
| Phase 1 source text preserved | 100.0% (2/2) |  |
| Image reference retained | 100.0% (2/2) |  |
| Page retained | 50.0% (2/4) |  |
| Capture status as expected | 60.0% (3/5) |  |
| Equations captured or appropriately flagged | 60.0% (3/5) |  |

### Operations

| Metric | Result | Note |
|---|---|---|
| Extractions (passage x category, plus relationship steps) | 17 |  |
| Model calls | 25 |  |
| Retries per extraction | 47.1% (8/17) |  |
| Failed extractions | 0.0% (0/17) |  |
| Extractions needing review | 64.7% (11/17) |  |
| Model time (sum of extractions) | 196.3 s |  |
| Model time per passage | 39.3 s |  |
| Wall-clock duration of the runs | 196.9 s |  |
| Input tokens | 86,735 |  |
| Output tokens | 25,439 |  |
| Estimated cost | n/a |  |

### Required units not found

| Passage | Unit | Category | Names | Seems merged into |
|---|---|---|---|---|
| p12-ebm | training | principle | Training of an EBM; EBM training | Energy function used only for inference |
| p14-gibbs | prob-special-case | principle | Probabilistic models as a special case of EBMs; Energies as unnormalized negative log probabilities |  |

14 reference unit(s) belong to a category that was not extracted on their passage and are not scored.

### Extracted units without a reference match

| Passage | Category | Title | Kind | Counted against precision |
|---|---|---|---|---|
| p11-roadmap | concept | joint embedding predictive architectures (JEPAs) | redundant of p11-roadmap/jepa | yes |
| p11-roadmap | concept | energy-based models (EBMs) | unmatched | yes |
| p13-training | principle | Energy-guided model | unmatched | no — needs a reviewer |
| p12-ebm | concept | energy function | redundant of p12-ebm/energy-inference-only | yes |
| p13-training | principle | Energy-based models: decisions as choosing the option with the highest score | redundant of p13-training/score-not-probability | yes |
| p13-training | principle | Energy-based models give more flexibility than probabilistic models | redundant of p13-training/ebm-flexibility | yes |
| p13-training | principle | label bias problem | redundant of p13-training/score-not-probability | yes |
| p13-training | principle | Loss functions in probabilistic setting are tied to negative log-likelihood | redundant of p13-training/nll-training | yes |

### Matched pairs for review

| Passage | Reference | Extracted title | Recognised by | Facts present | Missing facts | Source | Pages |
|---|---|---|---|---|---|---|---|
| p10-ssl | ebm-solution | energy-based models (EBMs) | name | 100% | — | correct | ok |
| p10-ssl | hjepa | hierarchical joint embedding predictive architecture (H-JEPA) | name | 100% | — | correct | ok |
| p10-ssl | ssl | self-supervised learning (SSL) | name | 100% | — | correct | ok |
| p11-roadmap | jepa | hierarchical joint embedding predictive architecture (H-JEPA) | name | 100% | — | correct | ok |
| p12-ebm | decision-energy | Energy-guided model decision principle | name | 100% | — | correct | ok |
| p12-ebm | ebm | energy-based models (EBMs) | name | 100% | — | correct | ok |
| p12-ebm | energy-inference-only | Energy function used only for inference | name | 100% | — | correct | ok |
| p12-ebm | inference | energy function | content only: read it | 100% | — | correct | ok |
| p12-ebm | multimodal | Multimodal dependency representation advantage of EBMs | content only: read it | 100% | — | correct | ok |
| p12-ebm | prob-normalization | Probabilistic model normalization limitation | name | 100% | — | correct | ok |
| p12-ebm | smooth | Smoothness of energy function for continuous y | content only: read it | 100% | — | correct | ok |
| p13-training | drawbacks | Advantage of EBMs in high dimensions versus loss of probabilistic setting | content only: read it | 100% | — | correct | ok |
| p13-training | ebm-flexibility | energy-based models (EBMs) | content only: read it | 100% | — | correct | ok |
| p13-training | label-bias | label bias problem | name | 100% | — | correct | ok |
| p13-training | nll-training | Training with probabilistic models | name | 100% | — | correct | ok |
| p13-training | score-not-probability | Choosing the highest-scoring option | name | 100% | — | correct | ok |
| p14-gibbs | gibbs | Gibbs-Boltzmann distribution | name | 100% | — | correct | ok |
| p14-gibbs | partition | partition function | name | 100% | — | correct | ok |
| p15-latent | latent-variables | latent variables | name | 100% | — | correct | ok |
| p15-latent | structured-prediction | structured prediction problems | name | 100% | — | correct | ok |

### Equations

| Passage | Reference | Found | Status | Original kept | Source kept | Image | Page | Flagged | Handled correctly |
|---|---|---|---|---|---|---|---|---|---|
| p12-ebm | argmin-y | yes (read while extracting a neighbouring passage) | n/a | yes | n/a | n/a | n/a | yes | no |
| p13-training | eq-nll | yes | incomplete | yes | yes | yes | yes | yes | yes |
| p14-gibbs | eq-gibbs | yes | needs_review | yes | yes | yes | yes | yes | yes |
| p15-latent | eq-argmin-z | yes (read while extracting a neighbouring passage) | needs_review | yes | n/a | n/a | no | yes | no |
| p15-latent | eq-marginal | yes (read while extracting a neighbouring passage) | needs_review | yes | n/a | n/a | no | yes | yes |
| p15-latent | eq-ellipse | not scored: equations were not extracted on this passage | | | | | | | |

### Relationships

| Passage | Expected | Required | Found | Stored type | Type accepted |
|---|---|---|---|---|---|
| p13-training | eq-nll → nll-training (describes/related_to) | yes | no |  |  |
| p14-gibbs | eq-gibbs → gibbs (describes/related_to) | yes | yes | describes | yes |
| p14-gibbs | partition → gibbs (part_of/related_to/depends_on) | yes | no |  |  |

Stored relationships the reference does not list:

| Source | Type | Target | Origin | Why not scored |
|---|---|---|---|---|
| label bias problem | related_to | label bias problem | extracted | touches a unit with no reference match |
| Training with probabilistic models | related_to | Loss functions in probabilistic setting are tied to negative log-likelihood | extracted | touches a unit with no reference match |
| Advantage of EBMs in high dimensions versus loss of probabilistic setting | related_to | energy-based models (EBMs) | extracted | both units recognised, pair not in the reference |
| Energy-based models: decisions as choosing the option with the highest score | applies_to | energy-based models (EBMs) | extracted | touches a unit with no reference match |
| label bias problem | related_to | Energy-based models: decisions as choosing the option with the highest score | extracted | touches a unit with no reference match |


## How to read this report

* The reference was prepared before any extraction was inspected, but it is one reviewer's reading. Where its status is `draft` it still needs a second, human review.
* Matching is lexical: names, key words and symbols. It shows whether expected knowledge is present, correctly placed and traceable. It cannot prove that a paraphrase keeps the author's meaning; the matched pairs below are listed for a reviewer to read.
* Precision counts unmatched units only in passages whose reference is exhaustive; elsewhere they are listed for judgement, not scored.
* A value of `n/a` means the denominator was zero or the provider did not report the figure. It is never shown as zero.
* Small datasets give coarse ratios: read the counts next to every percentage.


## Metric definitions

| Metric | Counting method |
|---|---|
| Required units found (recall) | Required reference units (not equations) that an observed unit matches, among those whose own category was extracted on their passage (a unit found under an accepted alternative category still counts). |
| Recall: concept | Required concept units found. |
| Recall: method | Required method units found. |
| Recall: principle | Required principle units found. |
| Recall in passages with several required units | Required units found, counting only passages whose reference lists two or more. |
| Extracted units explained by the reference (precision) | Units matching a reference unit (required or optional) divided by those plus redundant units and unmatched units in passages whose reference is exhaustive. |
| Passages without knowledge left without unexpected units | Passages the reviewer marked as holding no required knowledge in which every extracted unit is one the reference accepts (usually: none). |
| Required units that seem merged into another unit | Missed required units whose first name appears in the text of a unit matched to a different reference unit of the same passage. |
| Key facts present in matched units (lexical proxy) | Reference facts (each satisfied by any of its alternative words) found in the text of the matched unit. Presence of words, not proof of correct meaning. |
| Matched units containing a phrase the reviewer ruled out | Matched units whose text contains a forbidden phrase of their reference unit. |
| Units the pipeline flagged as unsupported | Units with an open unsupported-claim or quote-mismatch flag. |
| Matched units citing the right passage | Matched units whose primary source is the reference passage. |
| Matched units with pages inside the passage's pages | Matched units with page information whose range lies within the reference pages of the passage. |
| Units whose sources verified | Units with source_status 'verified' (passage exists, quotes found in the passage, pages inside it). |
| Units whose primary source is the passage that was extracted | Units whose primary source is the passage being extracted (the prompt says to extract only from it). A unit that cites a neighbouring passage as its primary source was read from context. |
| Quotes found in their passage | Source quotes whose words were found in the cited passage text. |
| Answers that matched their schema | Model answers (not relationship calls; failed provider calls have no answer) that parsed and matched the Pydantic schema, counting every retry. |
| Extractions that succeeded on the first call | Executions (passage and category) finished with a single call and not failed. |
| Matched units in the expected category | Matched units whose category is the reference's primary category (not only an accepted alternative). |
| Redundant units among stored units | Stored units that match a reference unit already matched by a better unit of the same passage, over all stored units in the dataset. |
| Duplicates the pipeline refused to store | Duplicate units skipped by the pipeline during the evaluated runs. |
| Required relationships found | Required reference relationships whose two units were both found, and that a stored relationship connects (either direction for symmetric types). |
| Found relationships of an accepted type | Of the found relationships, those whose type the reviewer accepts. |
| Relationships between recognised units that the reference expects | Stored relationships between two recognised units that the reference lists. Relationships touching unrecognised units cannot be judged and are listed separately. |
| Reference equations found | Reference equations for which an equation unit of the same passage holds at least 60% of the expected symbols. |
| Original expression preserved | Found equations whose original_expression holds at least 80% of the expected symbols. |
| Phase 1 source text preserved | Found equations whose stored source text or LaTeX is the Phase 1 one (never the model's). |
| Image reference retained | Found equations that should have an image whose crop reference is kept. |
| Page retained | Found equations whose source page lies within the reference pages. |
| Capture status as expected | Found equations whose stored capture status is one the reviewer accepts. |
| Equations captured or appropriately flagged | Reference equations that were found with an accepted status, and flagged when the reviewer said the source is incomplete or ambiguous. Equations that are missing count against. |
| Extractions (passage x category, plus relationship steps) | Executions recorded by the evaluated runs. |
| Model calls | Calls to the model, including retries. |
| Retries per extraction | Retries (corrections and resends) divided by executions. |
| Failed extractions | Executions that ended failed. |
| Extractions needing review | Executions that ended needs_review. |
| Model time (sum of extractions) | Sum of the recorded duration of every execution. |
| Model time per passage | Model time divided by the passages extracted. |
| Wall-clock duration of the runs | Sum of the runs' start-to-finish time. |
| Input tokens | Tokens sent, as reported by the provider (unavailable if it reports none). |
| Output tokens | Tokens received, as reported by the provider. |
| Estimated cost | Estimated from the configured token prices; unavailable if the model has no prices. |
