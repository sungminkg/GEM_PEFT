import numpy as np
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_f1_score(prediction, gold_answers):
    if gold_answers[0] == "CANNOTANSWER" or gold_answers[0] == "no answer":
        return int(normalize_answer(gold_answers[0]) == normalize_answer(prediction))

    all_f1s = []
    prediction_tokens = normalize_answer(prediction).split()
    for answer in gold_answers:
        ground_truth_tokens = normalize_answer(answer).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            all_f1s.append(0)
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        all_f1s.append((2 * precision * recall) / (precision + recall))

    return max(all_f1s)


def calculate_metric(predictions, metric_name):
    if metric_name == "accuracy":
        if isinstance(predictions[0].correct_candidate, list):
            return np.mean([pred.predicted_candidate in pred.correct_candidate for pred in predictions])
        else:
            return np.mean([pred.correct_candidate == pred.predicted_candidate for pred in predictions])

    elif metric_name == "em":
        # For question answering (string exact match)
        return np.mean([
            any([normalize_answer(ans) == normalize_answer(pred.predicted_candidate) for ans in pred.correct_candidate])
            for pred in predictions
        ])

    elif metric_name == "f1":
        return np.mean([
            qa_f1_score(pred.predicted_candidate, pred.correct_candidate)
            for pred in predictions
        ])

    elif metric_name == "exact_match":
        def extract_numeric_answer(s: str) -> str:
            s = s.lower()
            after = ""
            if "####" in s:
                after = s.split("####")[-1]
            elif "answer is" in s:
                after = s.split("answer is")[-1]
            elif "answer:" in s:
                after = s.split("answer:")[-1]
            else:
                after = s.strip().split('\n')[-1]

            match = re.search(r"-?\d[\d,]*\.?\d*", after)
            if match:
                return match.group(0).replace(",", "")

            all_match = re.findall(r"-?\d[\d,]*\.?\d*", s)
            if all_match:
                return all_match[-1].replace(",", "")

            return None

        def safe_float(x):
            try:
                return float(x)
            except:
                return None

        pred_answers = [extract_numeric_answer(pred.predicted_candidate) for pred in predictions]
        gold_answers = [extract_numeric_answer(pred.correct_candidate) for pred in predictions]

        matches = [
            safe_float(p) == safe_float(g) if p is not None and g is not None else False
            for p, g in zip(pred_answers, gold_answers)
        ]

        exact_match = sum(matches) / len(matches) if len(matches) > 0 else 0.0

        print("Extracted predicted answers (sample):", pred_answers[:3])
        print("Extracted gold answers (sample):", gold_answers[:3])
        print("Exact match:", exact_match)

        return exact_match
    
    else:
        raise ValueError(f"Unsupported metric name: {metric_name}")
