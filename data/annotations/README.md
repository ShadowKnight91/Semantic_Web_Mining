# Annotation guide (X1: extraction quality)

Each line of `annotator_A.jsonl` / `annotator_B.jsonl` is one call. Read the transcript in
`sample.jsonl` and fill `triples` with every fact the CALLER states, using only the ontology:

| relation | object type | object |
|---|---|---|
| CLAIMS_TO_BE | ClaimedIdentity | one allowed value, e.g. `government_agent`, `bank`, `tech_support` |
| MENTIONS_ORG | Organization | the organization name, e.g. `Social Security Administration` |
| USES_TACTIC | PressureTactic | `urgency`, `threat`, `reward`, `fear`, `secrecy`, `other` |
| REQUESTS | RequestedAsset | `otp`, `ssn`, `card_number`, `bank_account`, `password`, `remote_access`, `money`, `personal_info`, `other` |
| PAY_VIA | PaymentChannel | `gift_card`, `wire_transfer`, `crypto`, `zelle`, `p2p_app`, `bank_transfer`, `cash`, `check_or_money_order`, `credit_card_payment`, `other` |
| PROVIDES_NUMBER | CallbackNumber | the number as given, e.g. `1-888-555-0142`, `press 1` |

Example line:
```json
{"call_id": "bothbosu_00012", "annotator": "A", "triples": [
  {"relation": "CLAIMS_TO_BE", "object": "government_agent"},
  {"relation": "USES_TACTIC", "object": "threat", "inferred": false}]}
```
Full allowed values and descriptors: `ontology/ontology.yaml`. Annotate independently; afterwards
adjudicate disagreements into `gold.jsonl` (same format). `python -m scamkg eval-extraction`
reports agreement and scores every extraction run.
