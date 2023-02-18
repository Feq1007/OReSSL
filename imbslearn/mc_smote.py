from collections import Counter
from river import base, utils
from river.imblearn.random import ClassificationSampler
from imblearn.over_sampling import BorderlineSMOTE, ADASYN

class SMOTESampler(ClassificationSampler):
    def __init__(self, classifier: base.Classifier, seed: int = None, ir: int = 10):
        super().__init__(classifier, seed)
        self.ir_threshold = ir
        self.mc_counter = self.classifier.counter
        self.true_counter = Counter()
        self.generate_counter = Counter()

    def learn_one(self, x, y, **kwargs):
        if not self.classifier.initialized:
            return self.classifier.learn_one(x, y, kwargs)

        y_pred = kwargs['y_pred']
        re = max(y_pred.values())

        self.classifier.learn_one(x, y, y_pred)

        y = y if y != -1 else max(y_pred, key=y_pred.get)
        sample_size = int(max(self.mc_counter.values()) / self.ir_threshold) - self.mc_counter[y]
        if sample_size <= 0:
            return self

        x_sample, y_sample = self.borderline_smote_samples(x, y, sample_size=sample_size)
        y_pred = {c: 0.0 for c in self.classifier.classes}
        y_pred[y] = self.classifier.re_threshold
        for x_s, y_s in zip(x_sample, y_sample):
            t = {}
            for i, k in enumerate(x.keys()):
                t[k] = x_s[i]
            self.classifier.learn_one(t, y, y_pred)
        return self


    def borderline_smote_samples(self, x, y, sample_size=0):
        t = self.classifier.timestamp
        X, Y = [], []
        s = 0
        for mc in self.classifier.labeled_micro_clusters:
            if y == mc.label:
                s += 1
            center = mc.calc_center(t)
            X.append(list(center.values()))
            Y.append(mc.label)
        sm = BorderlineSMOTE(sampling_strategy={y:max(sample_size, s)}, k_neighbors=min(5, s))
        X_res, Y_res = sm.fit_resample(X, Y)
        return X_res, Y_res