"""Match-score decision logic.

Kept identical to FFAA's ``mids/selector.py`` so the upgraded model is a true drop-in:
the 4-class logits produced by MIDS++ are consumed by exactly the same selection rule used
by FFAA at inference time.

Class convention (matches upstream): for a candidate answer claiming ``real`` the relevant
classes are 0 (real image / real claim) and 2 (fake image / real claim); for a ``fake`` claim
they are 3 (fake image / fake claim) and 1 (real image / fake claim).  The match score is the
softmax-normalised probability that the answer's claim agrees with the true image authenticity.
"""

from __future__ import annotations

import torch


def make_decision(answers_result, scores):
    """Select the candidate answer with the highest match score.

    Args:
        answers_result: list of ``'real'``/``'fake'`` strings, one per candidate answer.
        scores: ``(num_answers, 4)`` softmax probabilities.

    Returns:
        ``(best_answer_idx, pred, match_score, forgery_score)`` where ``pred`` is the binary
        image-authenticity decision (0 real, 1 fake) and ``forgery_score`` is P(fake).
    """
    match_scores = []
    preds = []
    for i in range(len(answers_result)):
        if answers_result[i] == "real":
            match_scores.append(scores[i][0].item() / (scores[i][0].item() + scores[i][2].item()))
            preds.append(0)
        else:
            match_scores.append(scores[i][3].item() / (scores[i][3].item() + scores[i][1].item()))
            preds.append(1)

    scores = torch.tensor(match_scores, device=scores.device)
    best_answer_idx = torch.argmax(scores).item()
    pred = preds[best_answer_idx]
    match_score = match_scores[best_answer_idx]
    forgery_score = match_score if pred == 1 else 1 - match_score
    return best_answer_idx, pred, match_score, forgery_score


def make_decision9(claim_types, scores):
    """9-class (3 true-type x 3 claim-type) generalization of make_decision.

    Label convention: ``label = 3*true + claim`` with true/claim in {0 real, 1 pad, 2 deepfake}.
    For a candidate whose answer claims type ``c``, the match score is P(image truly is ``c`` | the
    answer claims ``c``) computed within that claim's column {3*0+c, 3*1+c, 3*2+c}. The most
    consistent candidate wins; its claim becomes the predicted type. Binary authenticity = real iff
    the winning claim is real (pad/deepfake both -> fake), so PAD+deepfake evidence pools toward fake.

    Args:
        claim_types: list of int claim indices (0 real, 1 pad, 2 deepfake), one per candidate.
        scores: ``(num_answers, 9)`` softmax probabilities.

    Returns:
        ``(best_idx, pred, match_score, forgery_score, pred_type)`` where ``pred`` is binary
        (0 real, 1 fake) and ``pred_type`` is the winning 3-way type.
    """
    match_scores = []
    for i in range(len(claim_types)):
        c = int(claim_types[i])
        col = scores[i][c].item() + scores[i][3 + c].item() + scores[i][6 + c].item()
        agree = scores[i][3 * c + c].item()
        match_scores.append(agree / col if col > 0 else 0.0)

    scores_t = torch.tensor(match_scores, device=scores.device)
    best_answer_idx = torch.argmax(scores_t).item()
    pred_type = int(claim_types[best_answer_idx])
    match_score = match_scores[best_answer_idx]
    pred = 0 if pred_type == 0 else 1
    forgery_score = match_score if pred == 1 else 1 - match_score
    return best_answer_idx, pred, match_score, forgery_score, pred_type


_CLAIM_IDX = {"real": 0, "pad": 1, "deepfake": 2, "fake": 2}


def make_decision9_batch(answers_result, scores, chunk_size=3):
    """Batched 9-class decision. ``answers_result`` are claim-type strings (real/pad/deepfake)."""
    chunks = [answers_result[i:i + chunk_size] for i in range(0, len(answers_result), chunk_size)]
    best_idxs, preds, match_scores, forgery_scores = [], [], [], []
    for b in range(len(chunks)):
        claim_types = [_CLAIM_IDX.get(r, 0) for r in chunks[b]]
        bi, pred, ms, fs, _ = make_decision9(claim_types, scores[b])
        best_idxs.append(bi); preds.append(pred); match_scores.append(ms); forgery_scores.append(fs)
    return best_idxs, preds, match_scores, forgery_scores


def make_decision_batch(answers_result, scores, chunk_size=3):
    answers_result = [answers_result[i:i + chunk_size] for i in range(0, len(answers_result), chunk_size)]

    best_answer_idxs = []
    preds = []
    match_scores = []
    forgery_scores = []

    for b in range(len(answers_result)):
        best_answer_idx, pred, match_score, forgery_score = make_decision(answers_result[b], scores[b])
        best_answer_idxs.append(best_answer_idx)
        preds.append(pred)
        match_scores.append(match_score)
        forgery_scores.append(forgery_score)
    return best_answer_idxs, preds, match_scores, forgery_scores
