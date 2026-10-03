from scamkg.ontology.standardize import EntityStandardizer, normalize_phone


def test_organization_aliases_merge(ontology):
    s = EntityStandardizer(ontology)
    names = {s.standardize("Organization", m) for m in ("SSA", "Social Security", "the social security office",
                                                        "Social Security Administration")}
    assert names == {"Social Security Administration"}
    assert s.standardize("Organization", "IRS") == "Internal Revenue Service"
    assert s.standardize("Organization", "Riverside Dental Clinic") == "Riverside Dental Clinic"
    assert s.standardize("Organization", "the bank") is None  # generic, not a named organization


def test_closed_vocabularies(ontology):
    s = EntityStandardizer(ontology)
    assert s.standardize("PaymentChannel", "Google Play gift cards") == "gift_card"
    assert s.standardize("PaymentChannel", "Bitcoin ATM") == "crypto"
    assert s.standardize("ClaimedIdentity", "Officer Brown from the Social Security Administration") == "government_agent"
    assert s.standardize("ClaimedIdentity", "Microsoft technician") == "tech_support"
    assert s.standardize("PressureTactic", "threat (arrest)") == "threat"
    assert s.standardize("RequestedAsset", "one-time password") == "otp"
    assert s.standardize("RequestedAsset", "something unusual") == "other"
    assert s.standardize("RequestedAsset", "none") is None


def test_retyping_repair(ontology):
    s = EntityStandardizer(ontology)
    assert s.infer_type("gift card", exclude="RequestedAsset") == ("PaymentChannel", "gift_card")


def test_phone_normalization():
    assert normalize_phone("(888) 555-0142") == "888-555-0142"
    assert normalize_phone("+1 888 555 0142") == "888-555-0142"
    assert normalize_phone("press 1") == "ivr:press-1"
    assert normalize_phone("extension 42") == "ext-42"
    assert normalize_phone("no number here") is None
