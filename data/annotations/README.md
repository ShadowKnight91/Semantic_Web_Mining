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
{"call_id": "bothbosu_00012", "annotator": "A", "done": true, "triples": [
  {"relation": "CLAIMS_TO_BE", "object": "government_agent"},
  {"relation": "USES_TACTIC", "object": "threat", "inferred": false}]}
```
Set `"done": true` on every call you have annotated - also when the caller states no ontology facts
at all (e.g. a wrong number): such a call stays a gold example with no triples, so any triple an
extractor invents for it counts as a false positive. Calls with `"done": false` and no triples are
treated as not yet annotated. Full allowed values and descriptors: `ontology/ontology.yaml`.

Decision rules (used for annotator A; follow them so agreement measures the extractors, not the rules):
* CLAIMS_TO_BE - who the caller presents as. "Officer X from the SSA" = `government_agent`; a caller from an
  insurance company = `insurance`; other sellers = `telemarketer_sales`; a prize / gift-card giveaway =
  `prize_or_lottery` (if the caller also claims another type, e.g. "we're a marketing firm", annotate both).
  A private person (wrong number, a customer calling a company) gets no identity; `relative_or_friend` only
  for the receiver's own family or friends.
* MENTIONS_ORG - every company or agency the caller names, placeholders ("XYZ Company") and brands (Dell)
  included; not generic references ("your bank"), not organizations only the receiver names, and "social
  security number" is not a mention of the SSA.
* USES_TACTIC - `urgency`: explicit time pressure or a deadline (today, immediately, within 48 hours, limited
  time); `threat`: arrest, legal action, suspension, fines, account closure; `reward`: refunds, prizes, free
  gifts, discounts, special deals (sales calls too); `fear`: suspicious activity, unauthorized charges, hacked /
  compromised accounts, viruses; `secrecy`: an explicit instruction to stay on the line or not to contact / call
  back the real organization. A delivery arriving "today" is not urgency.
* REQUESTS - only what the caller asks the receiver for: `ssn` (also the last four digits), `personal_info`
  (name, address, date of birth, email, phone number, policy details, "whose number is this?"),
  `card_number`, `bank_account`, `password`, `otp`, `remote_access` (also downloading a "support tool" or being
  talked through system settings - mark it inferred), `money` (fees, fines, payments).
* PAY_VIA - only how money is meant to move; a gift card that is the prize is not a payment channel, and a
  refund "deposited into your account" is `bank_transfer`.
* PROVIDES_NUMBER - phone numbers, extensions and "press N" options the caller gives; not badge, order,
  tracking or case numbers, and not websites.
* Mark `"inferred": true` when a fact is implied rather than stated.

Annotate independently; afterwards
adjudicate disagreements into `gold.jsonl` (same format). `python -m scamkg eval-extraction`
reports agreement and scores every extraction run.
