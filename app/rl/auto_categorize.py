# ai_categorizer.py
import os
import re
import joblib
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# --- Save model next to this script ---
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "transaction_categorizer.pkl")

try:
    import cloudpickle
except ImportError:
    cloudpickle = None

def save_model(model, model_path: str = MODEL_PATH):
    if cloudpickle is not None:
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)
    else:
        joblib.dump(model, model_path)

def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    if cloudpickle is not None:
        with open(model_path, "rb") as f:
            return cloudpickle.load(f)
    return joblib.load(model_path)

# ----------------------------
# Normalization / Cleanup
# ----------------------------

_BRAND_ALIASES = {
    r"\b(domino'?s|dominos)\b": "dominos",
    r"\b(youtube|google youtube|google\s+youtube)\b": "youtube",
    r"\b(hulu plus)\b": "hulu",
    r"\b(hello\s*fresh)\b": "hellofresh",
    r"\b(pay\s*pal|paypal)\b": "paypal",
    r"\b(venmo)\b": "venmo",
    r"\b(amazon\s*prime)\b": "amazonprime",
    r"\b(verizon wireless)\b": "verizon",
    r"\b(optimum cable)\b": "optimum",
    r"\b(usaa (?:insurance|p and c))\b": "usaa",
    r"\b(pizza\s*hutt?)\b": "pizzahut",
    r"\b(google pokemon)\b": "pokemon",
    r"\b(hidive)\b": "hidive",
    r"\b(playstation)\b": "playstation",
    r"\b(select media)\b": "selectmedia",
    r"\b(speedy stop)\b": "speedystop",
    r"\b(shogun japanese grill)\b": "shogun",
    r"\b(direct\s*deposit)\b": "directdeposit",
    r"\b(direct\s*deposit)\b": "directdeposit",
}

_STRIP_PATTERNS = [
    r"\b(card|recurring)\s+purchase(s)?(\s+with\s+pin)?\b",
    r"\brecurring\s+charge\b",
    r"\bsubscription\b",
    r"\bautopay\b",
    r"\bpayment\b",
    r"\btransfer\b",
    r"\bdraft\b",
    r"\bweb\s*id:\s*\S+\b",
    r"\bppd\s*id:\s*\S+\b",
    r"\bending\s+in\s+\d{3,4}\b",
    r"\b(e-?payment|inst(?:ant)?\s*x?fer)\b",
    r"\bcharge\b",
    r"\bcard\s*\d{3,4}\b",
    r"\b\d{2,}[-/]\d{1,}[-/]\d{1,}\b",   # dates like 2025-08-01
    r"\b#?\d{6,}\b",                     # long IDs
]

def _normalize_brands(s: str) -> str:
    for pat, repl in _BRAND_ALIASES.items():
        s = re.sub(pat, repl, s, flags=re.I)
    return s

def _strip_boilerplate(s: str) -> str:
    for pat in _STRIP_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.I)
    return s

def _clean_text(s: str) -> str:
    s = s.lower()
    s = _normalize_brands(s)
    s = _strip_boilerplate(s)
    s = re.sub(r"[^a-z0-9\s\.\-&/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------
# Keyword boost features
# ----------------------------

CATEGORY_KEYWORDS = {
    "Food": [
        "dominos","pizzahut","pizza","chipotle","mcdonald","starbucks","hellofresh",
        "grill","shogun","ubereats","doordash","restaurant","food","meal","burger","taco"
    ],
    "Entertainment": [
        "netflix","youtube","hulu", "hul*", "huluplus","playstation","hidive","selectmedia","fansly","patreon",
        "pokemon","spotify","disney","primevideo","crunchyroll","theater","cinema","xbox",
        "twitch","premium","media","subscription"
    ],
    "Transportation": [
        "uber","lyft","shell","chevron","exxon","valero","bp","texaco","speedystop",
        "carmax","fuel","gas","station","toll","metro","bus","train","taxi"
    ],
    "Shopping": [
        "amazonprime","amazon","target","bestbuy","walmart","ebay","membership","shop",
        "mall","costco","ikea","lowes","homedepot","retail","purchase", "target", "HEB"
    ],
    "Utilities": [
        "verizon","optimum","entergy","usaa","insurance","wireless","cable","spectrum",
        "comcast","att","at&t","t-mobile","internet","water","electric","gasbill","utility",
        "autopay","p&c","bill"
    ],
    "Housing": [
        "flex","flexfinance","rent","landlord","property","mortgage","housing","apartment",
        "lease","getflex","realestate","hoa"
    ],
    "Income": [
        "payroll","directdeposit","deposit","ltimindtree","paycheck","salary","bonus",
        "reimbursement","income","wages","pay","refund"
    ],
    "Debt Payment": [
        "discover","capitalone","chase","crcardpmt","e-payment","payment to","card payment",
        "credit card","loan","pmt","autopay","epayment"
    ],
    "Other": [
        "paypal","venmo","zelle","cashapp","transfer","instxfer","xfer","p2p","payment",
        "club","shaveclub","microsoft","misc","miscellaneous","unknown"
    ]
}

class KeywordBoost(BaseEstimator, TransformerMixin):
    def __init__(self, keywords_map: Dict[str, List[str]], kw_weight: float = 2.0):
        self.keywords_map = keywords_map
        self.kw_weight = kw_weight
        self.vec = DictVectorizer(sparse=True)
    def fit(self, X, y=None):
        feats = [self._extract_feats(text) for text in X]
        self.vec.fit(feats)
        return self
    def transform(self, X):
        feats = [self._extract_feats(text) for text in X]
        Xk = self.vec.transform(feats)
        return Xk * self.kw_weight
    def _extract_feats(self, text: str) -> Dict[str, int]:
        feats = {}
        for cat, kws in CATEGORY_KEYWORDS.items():
            hits = sum(1 for k in kws if k in text)
            feats[f"kw_count::{cat}"] = hits
            feats[f"kw_hit::{cat}"] = 1 if hits > 0 else 0
        return feats

# Make the class picklable from a stable module path (the filename without __main__)
KeywordBoost.__module__ = "app.rl.auto_categorize"


# ----------------------------
# Data helpers
# ----------------------------

def _to_df(data: Union[pd.DataFrame, List[Dict], Dict[str, List]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)
    if not {"description", "category"}.issubset(df.columns):
        raise ValueError("Data must include 'description' and 'category' columns")
    df = df.dropna(subset=["description", "category"])
    df["description"] = df["description"].astype(str).map(_clean_text)
    df["category"] = df["category"].astype(str)
    return df

def _pick_min_df(n_rows: int) -> int:
    # Avoid empty-vocab errors for tiny demos
    return 1 if n_rows < 200 else 2

# ----------------------------
# Train / Predict
# ----------------------------

CONF_THRESHOLD = 0.50  # tweak as you gather more data

def train_model(
    data,
    model_path: str = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[float, str]:
    df = _to_df(data)

    # Guards
    overall_classes = df["category"].unique()
    if len(overall_classes) < 2:
        raise ValueError("Need at least 2 distinct categories to train")
    if len(df) < 10:
        raise ValueError("Need at least 10 labeled rows to train reliably")

    # Stratify only if every class has >= 2 samples
    stratify = df["category"] if df["category"].value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["category"],
        test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Compute weights on classes that appear in y_train
    classes_in_train = np.unique(y_train)
    if len(classes_in_train) < 2:
        raise ValueError(
            "After splitting, the training set has <2 classes. "
            "Add more data, lower test_size, or ensure each class has at least 2 samples."
        )
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_in_train,
        y=y_train
    )
    weight_map = {c: w for c, w in zip(classes_in_train, class_weights)}
    sample_weight = y_train.map(weight_map).values
    min_per_class = y_train.value_counts().min()
    if min_per_class >= 3:
        # Safe to do 3-fold calibration
        try:
            calibrated_svc = CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced"),
                cv=3,
                method="sigmoid",
            )
        except TypeError:
            calibrated_svc = CalibratedClassifierCV(
                LinearSVC(class_weight="balanced"),
                cv=3,
                method="sigmoid",
            )
        final_clf = calibrated_svc

    elif min_per_class == 2:
        try:
            calibrated_svc = CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced"),
                cv=2,
                method="sigmoid",
            )
        except TypeError:
            calibrated_svc = CalibratedClassifierCV(
                LinearSVC(class_weight="balanced"),
                cv=2,
                method="sigmoid",
            )
        final_clf = calibrated_svc

    else:
        from sklearn.linear_model import LogisticRegression
        final_clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
            n_jobs=None,
        )

    model = Pipeline([
        ("features", FeatureUnion([
            ("w_tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=_pick_min_df(len(df)),
                sublinear_tf=True,
                strip_accents="unicode",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9&/.-]+\b",
                preprocessor=None,
            )),
            ("c_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=_pick_min_df(len(df)),
                strip_accents="unicode",
                preprocessor=None,
            )),
            ("kw_boost", KeywordBoost(CATEGORY_KEYWORDS)),
        ])),
        ("clf", final_clf),
    ])

    # Fit with sample weights
    model.fit(X_train, y_train, clf__sample_weight=sample_weight)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)

    # Save
    save_model(model, model_path)
    return acc, report

def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    return joblib.load(model_path)

def categorize(description: str, model: Optional[Pipeline] = None) -> str:
    if model is None:
        model = load_model()
    return model.predict([_clean_text(description)])[0]

def categorize_with_confidence(description: str, model: Optional[Pipeline] = None, top_k: int = 3):
    if model is None:
        model = load_model()
    desc = [_clean_text(description)]
    proba = model.predict_proba(desc)[0]
    classes = model.classes_
    idx = np.argsort(proba)[::-1]
    top = [(classes[i], float(proba[i])) for i in idx[:top_k]]
    pred, conf = top[0]
    label = pred if conf >= CONF_THRESHOLD else "Uncategorized"
    return label, conf, top

# ----------------------------
# Demo runner
# ----------------------------

if __name__ == "__main__":
    print("🔍 Starting AI Categorization Training...")

    data = [
      {"description": "Shogun Japanese Grill", "category": "Food"},
      {"description": "HelloFresh", "category": "Food"},
      {"description": "Dominos", "category": "Food"},
      {"description": "Pizza Hut", "category": "Food"},

      {"description": "Hulu", "category": "Entertainment"},
      {"description": "Netflix", "category": "Entertainment"},
      {"description": "HIDIVE", "category": "Entertainment"},
      {"description": "Patreon Inc", "category": "Entertainment"},
      {"description": "YouTube", "category": "Entertainment"},
      {"description": "Playstation", "category": "Entertainment"},
      {"description": "Google Pokemon", "category": "Entertainment"},
      {"description": "Select Media", "category": "Entertainment"},
      {"description": "Fansly", "category": "Entertainment"},

      {"description": "Speedy Stop", "category": "Transportation"},

      {"description": "Optimum Cable Payment", "category": "Utilities"},
      {"description": "Entergy Texas Bank Draft", "category": "Utilities"},
      {"description": "Verizon Wireless Payments", "category": "Utilities"},
      {"description": "USAA Insurance Autopay", "category": "Utilities"},

      {"description": "Flex Finance", "category": "Housing"},

      {"description": "Payroll Direct Deposit", "category": "Income"},
      {"description": "LTIMindtree Payroll", "category": "Income"},

      {"description": "Venmo Payment", "category": "Other"},
      {"description": "PayPal Instant Transfer", "category": "Other"},
      {"description": "Dollar Shave Club", "category": "Other"},
      {"description": "Payment To Chase Card", "category": "Other"},
      {"description": "Discover E-Payment", "category": "Other"},
      {"description": "Capital One Card Payment", "category": "Other"},
      {"description": "Amazon Prime", "category": "Shopping"},
      {"description": "PayPal Transfer Patreon Inc", "category": "Entertainment"},
      {"description": "PayPal Transfer HIDIVE", "category": "Entertainment"},
      {"description": "PayPal Transfer Google YouTube", "category": "Entertainment"},
      {"description": "PayPal Transfer Dominos", "category": "Food"},
      {"description": "PayPal Transfer Pizza Hut", "category": "Food"},
      {"description": "PayPal Transfer Google Pokemon", "category": "Entertainment"},
      {"description": "Recurring Charge Netflix", "category": "Entertainment"},
      {"description": "Recurring Charge Hulu Plus", "category": "Entertainment"},
      {"description": "Recurring Charge Dollar Shave Club", "category": "Other"},
      {"description": "Recurring Charge Flex Finance", "category": "Housing"},
      {"description": "Recurring Charge Verizon Wireless", "category": "Utilities"},
      {"description": "Recurring Charge Optimum", "category": "Utilities"},

      {"description": "Card Purchase Select Media", "category": "Entertainment"},
      {"description": "Card Purchase Fansly", "category": "Entertainment"},
      {"description": "Card Purchase Speedy Stop", "category": "Transportation"},
      {"description": "Auto Finance", "category": "Transportation"},
      {"description": "Card Purchase Shogun Japanese Grill", "category": "Food"},

      {"description": "Amazon Prime Membership", "category": "Shopping"},
      {"description": "Amazon Prime Charge", "category": "Shopping"},

      {"description": "Verizon Wireless Autopay", "category": "Utilities"},
      {"description": "USAA P and C Autopay", "category": "Utilities"},
      {"description": "Optimum Cable Autopay", "category": "Utilities"},
      {"description": "Entergy Texas Draft", "category": "Utilities"},

      {"description": "Chase Credit Card Payment", "category": "Debt Payment"},
      {"description": "Discover E-Payment", "category": "Debt Payment"},
      {"description": "Capital One Credit Card Payment", "category": "Debt Payment"},

      {"description": "Patreon Membership", "category": "Entertainment"},
      {"description": "YouTube Premium", "category": "Entertainment"},
      {"description": "Netflix Subscription", "category": "Entertainment"},
      {"description": "Hulu Subscription", "category": "Entertainment"},

      {"description": "HelloFresh Meal Kit", "category": "Food"},
      {"description": "Dominos Online Order", "category": "Food"},
      {"description": "Pizza Hut Online Order", "category": "Food"},

      {"description": "Venmo Transfer", "category": "Other"},
      {"description": "PayPal Inst Xfer", "category": "Other"}
    ]

    df = pd.DataFrame(data)

    try:
        accuracy, report = train_model(df)
        print("✅ Model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.3f}")
        print("📋 Classification Report:")
        print(report)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

    # --- Quick predictions ---
    model = load_model()
    tests = [
        "Chipotle order online",
        "SHELL OIL 1234",
        "Google YouTube subscription",
        "Domino's Pizza",
        "Uber trip downtown",
    ]
    for t in tests:
        label, conf, top = categorize_with_confidence(t, model)
        print(f"→ {t!r} => {label} (p={conf:.2f}) | top={top}")
