def init_score():
    scores = {"seen_err": 0,
              "seen_acc": 0,
              "seen_occ_err": 0,
              "seen_occ_acc": 0,
              "unseen_err": 0,
              "unseen_acc": 0,
              "unseen_occ_err": 0,
              "unseen_occ_acc": 0}
    return scores


def update_score(current_score, new_score):
    """
    Tracking best error and best accuracy
    """
    names = current_score.keys()
    for name in names:
        if name.endswith("err"):  # take the minimum
            current_score[name] = min(current_score[name], new_score[name])
        elif name.endswith("acc"):  # take the maximum
            current_score[name] = max(current_score[name], new_score[name])


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n