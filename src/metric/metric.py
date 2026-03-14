import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback


def normalize(text: str) -> str:
    """Lowercase, strip whitespace."""
    return text.strip().lower()


def f1_score(prediction: list[str], ground_truth: list[str]) -> float:
    pred_set = {normalize(a) for a in prediction}
    true_set = {normalize(a) for a in ground_truth}

    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _to_list(answer) -> list[str]:
    """Coerce an answer to a list of strings."""
    if isinstance(answer, list):
        return answer
    return [answer]


def phantomwiki_f1(gold, pred, trace=None):
    """Answer-level F1. Returns float for Evaluate."""
    gold_answers = _to_list(gold.answer)
    pred_answers = _to_list(getattr(pred, "answer", str(pred)))
    return f1_score(pred_answers, gold_answers)


def phantomwiki_f1_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Answer-level F1 with textual feedback for GEPA."""
    gold_answers = _to_list(gold.answer)
    pred_answers = _to_list(getattr(pred, "answer", str(pred)))
    score = f1_score(pred_answers, gold_answers)

    # Parse prediction into set for detailed feedback
    pred_set = {normalize(a) for a in pred_answers}
    true_set = {normalize(a) for a in gold_answers}
    correct = pred_set & true_set
    missed = true_set - pred_set
    extra = pred_set - true_set

    feedback = f"Gold answers ({len(true_set)}): {str(gold_answers)[:1000]}{'...' if len(str(gold_answers)) > 1000 else ''}. "
    feedback += f"Predicted ({len(pred_set)}): {str(pred_answers)[:1000]}. "
    feedback += f"F1: {score:.2f}. "
    if correct:
        feedback += f"Correct: {list(correct)[:5]}. "
    if missed:
        feedback += f"Missed: {list(missed)[:5]}{'...' if len(missed) > 5 else ''}. "
    if extra:
        feedback += f"Extra (wrong): {list(extra)[:5]}. "

    return ScoreWithFeedback(score=score, feedback=feedback)
