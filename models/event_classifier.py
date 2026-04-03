"""
Event Classification Model
===========================
Fine-tunes DistilBERT to classify geopolitical events into 5 categories.
Supports training, evaluation, inference, and batch prediction.

Categories:
  0 - Military/Conflict
  1 - Trade/Economic
  2 - Diplomatic
  3 - Humanitarian/Aid
  4 - Political
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Check GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LABEL_NAMES = ["Military/Conflict", "Trade/Economic", "Diplomatic", "Humanitarian/Aid", "Political"]
NUM_LABELS = len(LABEL_NAMES)


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_path: str = "models/event_classifier"
    eval_split: float = 0.15


def create_synthetic_training_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic training data for event classification.
    In production, replace with labeled GDELT + news data.
    """
    templates = {
        0: [  # Military/Conflict
            "{c1} launched military strikes against {c2} forces near the border",
            "{c1} troops clashed with {c2} soldiers in disputed territory",
            "{c1} deployed naval vessels in response to {c2} provocations",
            "Armed conflict escalates between {c1} and {c2} in northern region",
            "{c1} conducted airstrikes targeting {c2} infrastructure",
            "{c1} imposed a naval blockade on {c2} shipping lanes",
            "Casualties reported as {c1} and {c2} exchange fire along frontier",
        ],
        1: [  # Trade/Economic
            "{c1} and {c2} signed a comprehensive free trade agreement",
            "{c1} imposed tariffs on imports from {c2} citing dumping concerns",
            "{c2} became {c1}'s largest trading partner for the third year",
            "Investment flows between {c1} and {c2} reached a record high",
            "{c1} announced sanctions targeting {c2}'s energy sector",
            "Trade negotiations between {c1} and {c2} stalled over agriculture",
            "{c1} and {c2} agreed on joint infrastructure development projects",
        ],
        2: [  # Diplomatic
            "The {c1} ambassador was summoned to the {c2} foreign ministry",
            "{c1} and {c2} leaders held a bilateral summit in Geneva",
            "{c1} recognized {c2}'s territorial claims in the disputed region",
            "Diplomatic ties between {c1} and {c2} were fully restored",
            "{c1} recalled its ambassador to {c2} amid the crisis",
            "Foreign ministers of {c1} and {c2} signed a cooperation treaty",
            "{c1} formally protested {c2}'s decision at the UN Security Council",
        ],
        3: [  # Humanitarian/Aid
            "{c1} pledged emergency humanitarian aid to disaster-stricken {c2}",
            "Relief convoys from {c1} crossed into {c2} to assist refugees",
            "{c1} and {c2} launched a joint vaccination campaign",
            "{c1} donated medical supplies and food aid to {c2}",
            "Search and rescue teams from {c1} arrived to support {c2}",
            "{c1} opened its borders to refugees fleeing violence in {c2}",
            "Development assistance from {c1} will build infrastructure in {c2}",
        ],
        4: [  # Political
            "{c1} condemned {c2}'s recent election results as fraudulent",
            "{c1} expressed concern over the democratic backsliding in {c2}",
            "Political pressure from {c1} forced {c2} to reconsider its stance",
            "{c1} supported {c2}'s bid for UN Security Council membership",
            "{c1} and {c2} coordinated positions ahead of the G20 summit",
            "Opposition leaders from {c2} sought political asylum in {c1}",
            "{c1} and {c2} jointly proposed a resolution at the UN General Assembly",
        ]
    }

    countries = ["USA", "China", "Russia", "Germany", "UK", "France", "India",
                 "Brazil", "Japan", "South Korea", "Iran", "Saudi Arabia",
                 "Turkey", "Pakistan", "Israel", "Australia", "Canada"]

    records = []
    import random
    for _ in range(n):
        label = random.choice(list(templates.keys()))
        template = random.choice(templates[label])
        c1, c2 = random.sample(countries, 2)
        text = template.format(c1=c1, c2=c2)
        records.append({"text": text, "label": label})

    return pd.DataFrame(records)


class GeopoliticalEventClassifier:
    """
    DistilBERT-based event type classifier.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _load_dependencies(self):
        """Lazy load HuggingFace dependencies."""
        try:
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback,
            )
            from datasets import Dataset
            return True
        except ImportError:
            print("Install: pip install transformers datasets torch")
            return False

    def load_tokenizer_model(self, from_checkpoint: Optional[str] = None):
        """Load tokenizer and model (pretrained or from checkpoint)."""
        from transformers import (
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
        )

        model_path = from_checkpoint or self.config.model_name
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        ).to(DEVICE)

        self._loaded = True
        print(f"Model loaded from: {model_path}")

    def prepare_dataset(self, df: pd.DataFrame):
        """Tokenize and prepare HuggingFace Dataset."""
        from datasets import Dataset

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length
            )

        dataset = Dataset.from_pandas(df[["text", "label"]])
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    def train(self, df: Optional[pd.DataFrame] = None, eval_df: Optional[pd.DataFrame] = None):
        """
        Train the event classifier.

        Parameters
        ----------
        df : training DataFrame with 'text' and 'label' columns
        eval_df : optional evaluation DataFrame
        """
        from transformers import TrainingArguments, Trainer
        from sklearn.metrics import accuracy_score, f1_score
        import evaluate

        if df is None:
            print("No training data provided. Generating synthetic data...")
            df = create_synthetic_training_data(2000)

        if not self._loaded:
            self.load_tokenizer_model()

        # Split if no eval set
        if eval_df is None:
            split = int(len(df) * (1 - self.config.eval_split))
            eval_df = df.iloc[split:].reset_index(drop=True)
            df = df.iloc[:split].reset_index(drop=True)

        train_dataset = self.prepare_dataset(df)
        eval_dataset = self.prepare_dataset(eval_df)

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            acc = accuracy_score(p.label_ids, preds)
            f1 = f1_score(p.label_ids, preds, average="weighted")
            return {"accuracy": acc, "f1": f1}

        training_args = TrainingArguments(
            output_dir=self.config.save_path,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        print("Starting training...")
        trainer.train()

        # Save model and tokenizer
        os.makedirs(self.config.save_path, exist_ok=True)
        self.model.save_pretrained(self.config.save_path)
        self.tokenizer.save_pretrained(self.config.save_path)

        # Save label map
        with open(os.path.join(self.config.save_path, "label_map.json"), "w") as f:
            json.dump({str(i): name for i, name in enumerate(LABEL_NAMES)}, f)

        print(f"Model saved to {self.config.save_path}")
        return trainer

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict event types for a list of texts.

        Returns list of dicts: {label_id, label_name, confidence, all_scores}
        """
        if not self._loaded:
            if os.path.exists(self.config.save_path):
                self.load_tokenizer_model(self.config.save_path)
            else:
                # Rule-based fallback
                return [self._rule_based_classify(t) for t in texts]

        self.model.eval()
        results = []

        # Batch inference
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for p in probs:
                pred_idx = int(np.argmax(p))
                results.append({
                    "label_id": pred_idx,
                    "label_name": LABEL_NAMES[pred_idx],
                    "confidence": round(float(p[pred_idx]), 4),
                    "all_scores": {name: round(float(p[i]), 4)
                                   for i, name in enumerate(LABEL_NAMES)}
                })

        return results

    def _rule_based_classify(self, text: str) -> Dict:
        """
        Simple keyword-based fallback classifier.
        Used when the neural model hasn't been trained.
        """
        text_lower = text.lower()

        keywords = {
            0: ["attack", "military", "strike", "war", "conflict", "troops", "weapon",
                "bomb", "missile", "blockade", "invasion", "casualties"],
            1: ["trade", "tariff", "import", "export", "economic", "sanction", "investment",
                "gdp", "market", "commerce", "deal", "agreement"],
            2: ["diplomat", "summit", "ambassador", "treaty", "foreign minister",
                "negotiate", "recognize", "bilateral", "relations"],
            3: ["humanitarian", "aid", "refugee", "disaster", "relief", "food",
                "medical", "asylum", "donate", "support"],
            4: ["election", "democracy", "political", "un resolution", "vote",
                "government", "protest", "sanctions", "opposition"],
        }

        scores = {}
        for label_id, kws in keywords.items():
            scores[label_id] = sum(1 for kw in kws if kw in text_lower)

        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        conf = scores[best] / total

        return {
            "label_id": best,
            "label_name": LABEL_NAMES[best],
            "confidence": round(conf, 4),
            "all_scores": {LABEL_NAMES[i]: round(scores.get(i, 0) / total, 4)
                           for i in range(NUM_LABELS)},
            "method": "rule_based"
        }

    def predict_dataframe(self, df: pd.DataFrame, text_col: str = "SOURCEURL") -> pd.DataFrame:
        """
        Add predicted event type columns to a DataFrame.
        """
        if text_col not in df.columns:
            print(f"Column '{text_col}' not found. Using event_label instead.")
            text_col = "event_label" if "event_label" in df.columns else None

        if text_col is None or text_col not in df.columns:
            df["ml_event_type"] = "Unknown"
            df["ml_confidence"] = 0.0
            return df

        texts = df[text_col].fillna("").astype(str).tolist()
        preds = self.predict(texts)

        df = df.copy()
        df["ml_event_type"] = [p["label_name"] for p in preds]
        df["ml_confidence"] = [p["confidence"] for p in preds]
        return df


def build_training_dataset_from_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create labeled training data from GDELT event codes + descriptions.
    Maps CAMEO codes to event type labels.
    """
    cameo_to_label = {
        # Cooperation
        "03": 2, "04": 2, "05": 2,
        "06": 1, "07": 3,
        "01": 4, "02": 4, "08": 4,
        # Conflict
        "09": 4, "10": 4, "11": 4, "12": 4,
        "13": 0, "14": 4, "15": 0,
        "16": 2, "17": 0,
        "18": 0, "19": 0, "20": 0,
    }

    records = []
    for _, row in df.iterrows():
        code = str(row.get("EventRootCode", ""))
        label = cameo_to_label.get(code[:2], 4)  # default: political
        text = f"{row['Actor1CountryCode']} {row.get('event_label', '')} {row['Actor2CountryCode']}"
        records.append({"text": text, "label": label})

    return pd.DataFrame(records)
