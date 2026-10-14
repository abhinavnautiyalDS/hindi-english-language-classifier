from src.language_classifier import normalize_text, build_logistic_regression


def test_normalize_text():
    assert normalize_text("  HELLO\\n  THERE ") == "hello there"


def test_model_fits_and_predicts():
    texts = [
        "मुझे फीस बताइए", "मुझे admission चाहिए",
        "please call me tomorrow", "send me the details",
        "sir mujhe fees chahiye", "kal meeting hai please confirm",
    ]
    labels = ["hindi", "hinglish", "english", "english", "hinglish", "hinglish"]
    model = build_logistic_regression()
    model.fit(texts, labels)
    assert model.predict(["please call me"])[0] in {"hindi", "english", "hinglish"}
