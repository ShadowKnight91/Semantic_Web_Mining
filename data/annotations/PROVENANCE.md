# How the X1 gold annotations were made

Both annotation passes over the 50-call sample (`sample.jsonl`: 50 BothBosu calls, 25 scams, 25 legitimate)
were made with AI assistance, which the course instructor approved for this project.

| | Annotator A (`annotator_A.jsonl`) | Annotator B (`annotator_B.jsonl`) |
|---|---|---|
| Made by | Claude | Claude, a separate later pass |
| Input | the transcripts in `sample.jsonl`; the guide's decision rules were written down during this pass | the same transcripts and the finished guide (`README.md`) only |
| Blind to | not recorded | annotator A's labels and every extractor's output (A was opened only after B was written and frozen) |
| Triples | 202 on 50 calls | 202 written, 201 after standardization (`Walmart Rewards Inc.` and `Walmart` merge), 3 calls with no facts |

Agreement (B against A, `results/extraction_quality.json` -> `inter_annotator`): triple F1 0.993, precision 0.995,
recall 0.990; every relation agrees fully except `USES_TACTIC` (F1 0.983) and `REQUESTS` (0.987). The three
disagreements are all borderline cases of the guide's rules:

| Call | A only | B only | Transcript |
|---|---|---|---|
| `bothbosu_00067` | `USES_TACTIC urgency` | | "Just give me your social security number and we can fix this right now." |
| `bothbosu_01305` | `USES_TACTIC secrecy` | | to "Can I call you back later?": "Ma'am, I wouldn't advise that." |
| `bothbosu_00610` | | `REQUESTS card_number` (inferred) | to "before I give you my credit card information": "we'll take care of everything over the phone" |

**Read the agreement as an upper bound.** Both passes come from the same model family and follow the same
detailed decision rules, and the BothBosu dialogues state most facts explicitly, so this measures how
consistently the guide can be applied, not how much two people would disagree. A human check of a few calls
would make it stronger.

The extraction scores in X1 use annotator A as the reference. No `gold.jsonl` was adjudicated: resolving the
three disagreements would change at most three of A's 202 triples.
