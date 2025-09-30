from typing import List, Tuple
def rerank(items: List[Tuple[str, float]], p300_scores: dict, alpha=1.0):
    def s(item):
        iid, prior = item
        return prior + alpha*p300_scores.get(iid, 0.0)
    return sorted(items, key=s, reverse=True)
