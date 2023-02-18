import copy
import math
import operator
import typing
import river.base as base
import river.utils as utils

from abc import ABCMeta
from river.stats import Mean
from collections import Counter


class ReStream(base.Classifier):
    r"""ReStream

    **Online micro-cluster maintenance**

    For a new point 'p':
    
    * Step One: predict_prob_one
        * Compute the distance between 'p' and each micro cluster centriod. 
            * Method 1: Find all micro clusters for which 'p' falls within the fixed radius and use the infornation to decide the lable.
            * Method 2: Use KNN as base classifier to predict the label
              * 找到最近K个微簇，获取其距离以及对于微簇的可靠性，求解**可靠距离**，KNN进行分类后，累计同一类的可靠距离，并通过softmax进行放大。
              * 如果当前样本是有标签样本，则对进行分类的微簇更新其可靠性

        * Return the predict label and probability

    * Step Two: learn_one
        * Find all micro clusters for which 'p' falls within the dixed radius.
            * If no neighbor found. create a new micro cluster with a weight 1 by 'p'
            * Else we update the micro clusters by applying the appropriate fading, increasing their weight and then we try to move them closer to 'p' using the Gaussian neighborhood function.
              (如果是有标签样本，我们将相同类别的微簇往其靠近，并提高其可靠性)
              (如果是)

        * MyMethod:
            * 首先找到最近的微簇，如果样本在微簇中且样本可靠：
                * 进一步判断是否有标签，如果有标签，并且预测正确或者是无标签微簇，那么插入，并设置微簇标签和可靠性
                * 如果样本为无标签样本且预测标签和微簇标签一致或微簇无标签，也插入微簇，设置标签和可靠性
                * 否则需要创建新微簇，但是要保证能创建成功，所以需要判断是否满足维护个数要求，如果超过，需要删除部分微簇
                  然后创建微簇，根据
            

    """

    def __init__(
            self,
            knn: int = 30,  # k of KNN
            max_labeled_micro_clusters: int = 500,
            max_unlabeled_micro_clusters: int = 200,
            n_samples_init: int = 200,  # 用于初始化微簇的样本数
            cleanup_interval: int = 100,  # 进行周期处理的间隔（比如删除旧微簇）
            stream_speed: int = 1,  # 通过speed控制timestamp

            weighted: bool = True,  # 是否使用可靠性距离来预测
            softmax: bool = True,  # 预测结果是否用softmax归一化
            imbalance: bool = False,

            decaying_factor: float = 0.0025,  # 微簇插入数据时，历史数据的权重因子
            epsilon: float = 0.0001,      # 控制可靠性更新幅度

            re_threshold: float = 0.90,  # 可靠性阈值
            imb_threshold: float = 100,

            seed: int = 42,  # 生成随机中心位置（KMeans）
            **kwargs,
    ) -> None:
        super().__init__()
        self.timestamp = 0
        self._n_samples_seen = 0

        self.knn = knn
        self.seed = seed
        self.stream_speed = stream_speed
        self.n_samples_init = n_samples_init
        self.cleanup_interval = cleanup_interval
        self.max_labeled_micro_clusters = max_labeled_micro_clusters
        self.max_unlabeled_micro_clusters = max_unlabeled_micro_clusters

        self.weighted = weighted
        self.softmax = softmax

        self.re_threshold = re_threshold
        self.epsilon = epsilon

        self.decaying_factor = decaying_factor

        self.initialized = False

        self.classes: typing.Set[base.typing.ClfTarget] = set()
        self.labeled_micro_clusters: typing.List["ReMicroCluster"] = []
        self.unlabeled_micro_clusters: typing.List["ReMicroCluster"] = []

        # self._init_buffer: typing.Deque[typing.Dict] = collections.deque() # _init_kmeans()

        self.clustering_threshold = 1.5 # not used: 以x为中心寻找最近范围内的微簇

        self.counter = Counter()

        # temporary variable
        self.l_knearest = None
        self.u_nearest = None
        self.fix_neighbors = None

        # statistic info
        self.mean_radius = Mean()
        self.mean_func_radius = Mean()
        self.mean_re = Mean()
        self.mean_distance = Mean()
        self.drop_labeled = 0
        self.drop_unlabeled = 0
        self.labeled_num = 0
        self.insert_labeled = 0
        self.replace_labeled = 0
        self.insert_reliable = 0
        self.replace_reliable = 0
        self.change_labeled = 0
        self.change_reliable = 0
        self.insert_unlabeled = 0
        self.create_labeled = 0
        self.create_reliable = 0
        self.create_unreliable = 0
        self.create_unlabeled = 0


    @property
    def centers(self):
        return self.labeled_micro_clusters, self.unlabeled_micro_clusters

    def describe(self):
        print("======= ReStream Statistic Info ======")
        print(f"labeled_num  : {self.labeled_num}")
        print(f"seen number  : {self._n_samples_seen}")
        print(f"labeled micro cluster number  : {len(self.labeled_micro_clusters)}")
        print(f"unlabeled micro cluster number: {len(self.unlabeled_micro_clusters)}")
        print(f"mean_re  : {self.mean_re.get()}")
        print(f"mean_radius  : {self.mean_radius.get()}")
        print(f"mean_func_radius  : {self.mean_func_radius.get()}")
        print(f"mean_knn_distance  : {self.mean_distance.get()}")
        print(f"create_labeled  : {self.create_labeled}")
        print(f"create_reliable  : {self.create_reliable}")
        print(f"create_unreliable  : {self.create_unreliable}")
        print(f"create_unlabeled  : {self.create_unlabeled}")
        print(f"drop_labeled  : {self.drop_labeled}")
        print(f"drop_unlabeled  : {self.drop_unlabeled}")
        print(f"insert_labeled  : {self.insert_labeled}")
        print(f"replace_labeled  : {self.replace_labeled}")
        print(f"insert_reliable  : {self.insert_reliable}")
        print(f"replace_reliable  : {self.replace_reliable}")
        print(f"change_labeled  : {self.change_labeled}")
        print(f"change_reliable  : {self.change_reliable}")
        print(f"insert_unlabeled  : {self.insert_unlabeled}")
        c = {}
        for mc in self.labeled_micro_clusters:
            if mc.label not in c:
                c[mc.label] = 1
            else:
                c[mc.label] += 1
        print(f'class distribution: {c}')

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if not self.initialized:
            return {}

        # Return the k closest points (first index is MicroCluster & last index is distance)
        self.l_knearest = self._get_k_nearest_clusters(x, self.labeled_micro_clusters, self.knn)

        y_pred = {c: 0.0 for c in self.classes}

        # No nearest points? Return the default (normalized)
        if not self.l_knearest:
            default_pred = 1 / len(self.classes) if self.classes else 0.0
            return {c: default_pred for c in self.classes}

        # If the closest is exact the point x, return it
        if self.l_knearest[0][-1] == 0 and self.l_knearest[0][0].label is not None:
            y_pred[self.l_knearest[0][0].label] = 1.0
            return y_pred

        # statistic distance
        for neighbor in self.l_knearest:
            mc, distance = neighbor

            # weighted votes by inverse distance
            if self.weighted:
                y_pred[mc.label] += mc.re / distance

            # Uniform votes
            else:
                y_pred[mc.label] += 1.0

            self.mean_distance.update(distance)

        # Normalize votes into real [0, 1] probabilities
        if self.softmax:
            return utils.math.softmax(y_pred)

        # Otherwise normalized by the total sum
        total = sum(y_pred.values())
        for y in y_pred:
            y_pred[y] /= total
        return y_pred

    def insert_directly(self, x, y, re=1, labeled=True):
        if labeled:
            mc_from_x = ReMicroCluster(
                x=x,
                label=y,
                re=1,
                timestamp=self.timestamp,
                decaying_factor=self.decaying_factor)
            self.labeled_micro_clusters.append(mc_from_x)
        else:
            mc_from_p = ReMicroCluster(
                x=x,
                label=-1,
                re=re,
                timestamp=self.timestamp,
                decaying_factor=self.decaying_factor,
            )
            self.unlabeled_micro_clusters.append(mc_from_p)

    def _calc_radius(self, closest_mc, labeled=True):
        # Check whether the new instance fits into the closest micro-cluster
        if labeled:
            micro_clusters = self.labeled_micro_clusters
        else:
            micro_clusters = self.unlabeled_micro_clusters

        if closest_mc.N == 1:
            radius = math.inf
            center = closest_mc.calc_center(self.timestamp)
            for mc_id, mc in enumerate(micro_clusters):
                if mc == closest_mc:
                    continue
                distance = self._distance(mc.calc_center(self.timestamp), center)
                radius = min(distance, radius)
        else:
            radius = closest_mc.calc_radius(self.timestamp)
            self.mean_radius.update(radius)
        self.mean_func_radius.update(radius)
        return radius

    def _maintain_micro_clusters(self, ):
        # Delete old micro-cluster if its relevance stamp is smaller than the threshold
        for i, mc in enumerate(self.labeled_micro_clusters):
            if mc.update(self.timestamp, self.epsilon) < self.re_threshold:
                self.counter[mc.label] -= 1     # update statistic information
                self.drop_labeled += 1
                self.labeled_micro_clusters.pop(i)

        for j, mc in enumerate(self.unlabeled_micro_clusters):
            if mc.update(self.timestamp, self.epsilon) < self.re_threshold:
                self.drop_unlabeled += 1
                self.unlabeled_micro_clusters.pop(j)

    def learn_one(self, x, y, y_pred=None, sample_weight=None):
        self._n_samples_seen += 1
        if self._n_samples_seen % self.stream_speed == 0:
            self.timestamp += 1

        # Initialization
        if not self.initialized:
            if len(self.labeled_micro_clusters) < self.n_samples_init - 1:
                self.timestamp = 0
                self.insert_directly(x, y)
                self.classes.add(y)
                self.counter[y] += 1
                return self
            else:
                self.initialized = True
                return self

        # Adjust the reliability according to the true label
        if self._is_labeled(y):
            # Update
            self._update(y, y_pred)

        # Insertion
        self._insert(x, y, y_pred)

        # maintain micro clusters's reliability
        self._maintain_micro_clusters()

        # Periodic cluster remova
        # if self.timestamp > 0 and self.timestamp % self.cleanup_interval == 0:
        #     print(len(self.labeled_micro_clusters),len(self.unlabeled_micro_clusters))
        #     if len(self.labeled_micro_clusters) > self.max_labeled_micro_clusters:
        #         self._merge_two_closest_micro_cluster()
        #     if len(self.unlabeled_micro_clusters) > self.max_labeled_micro_clusters:
        #         self._merge_two_closest_micro_cluster(unlabeled=True)
        return self

    def _update(self, y, y_pred):
        self.labeled_num += 1
        if len(self.l_knearest) < 1 or y_pred==None:
            return

        for i, (mc, dis) in enumerate(self.l_knearest):
            if mc.label == y:
                mc.re += max(1 - mc.re, (1 - mc.re) * math.pow(math.e, y_pred.get(mc.label) - 1))
            else:
                mc.re -= (1.01 - mc.re) * math.pow(math.e, y_pred.get(mc.label))

    def _is_labeled(self, y):
        return y != -1

    def _insert(self, point, label, label_prob):
        re = max(label_prob.values())
        label_pred = max(label_prob, key=label_prob.get)
        self.mean_re.update(re)

        out = True
        self.u_nearest = self._get_closest_cluster_key_dis(point, self.unlabeled_micro_clusters)
        # find the nearest microcluster
        if self.u_nearest[0] == -1 or self.l_knearest[0][-1] < self.u_nearest[-1]:  # nearest is labeled
            if self.l_knearest[0][-1] < self._calc_radius(self.l_knearest[0][0],
                                                          labeled=True):  # in the microcluster's area
                out = False
                if label != -1 and self.l_knearest[0][0].label == label:
                    self.insert_labeled += 1
                    self.l_knearest[0][0].insert(point, label, 1, self.timestamp)
                elif label != -1 and self.l_knearest[0][0].label != label:
                    self.counter[self.l_knearest[0][0].label] -= 1      # update counter
                    self.counter[label] += 1                            # update counter
                    self.replace_labeled += 1
                    self.l_knearest[0][0].replace(point, label, 1, self.timestamp)
                elif label == -1 and re > self.re_threshold and self.l_knearest[0][0].label == label_pred:
                    self.insert_reliable += 1
                    self.l_knearest[0][0].insert(point, label_pred, re, self.timestamp)
                elif label == -1 and re > self.re_threshold and self.l_knearest[0][0].label != label_pred:
                    self.counter[self.l_knearest[0][0].label] -= 1      # update counter
                    self.counter[label_pred] += 1
                    self.replace_reliable += 1
                    self.l_knearest[0][0].replace(point, label_pred, re, self.timestamp)
                else:  # not labeled and not reliabe
                    self.insert_directly(point, -1, 1, labeled=False)
                    self.create_unreliable += 1
        else:
            if self.u_nearest[0] != -1 and self.u_nearest[-1] < self._calc_radius(
                    self.unlabeled_micro_clusters[self.u_nearest[0]], labeled=False):
                out = False
                updated_mc = copy.copy(self.unlabeled_micro_clusters[self.u_nearest[0]])
                if label != -1:
                    updated_mc.insert(point, label, 1, self.timestamp)
                    del self.unlabeled_micro_clusters[self.u_nearest[0]]
                    self.labeled_micro_clusters.append(updated_mc)
                    self.change_labeled += 1
                    self.counter[label] += 1                    # update counter
                elif re > self.re_threshold:
                    updated_mc.insert(point, label_pred, re, self.timestamp)
                    del self.unlabeled_micro_clusters[self.u_nearest[0]]
                    self.labeled_micro_clusters.append(updated_mc)
                    self.change_reliable += 1
                    self.counter[label_pred] += 1               # update counter
                else:
                    self.unlabeled_micro_clusters[self.u_nearest[0]].insert(point, -1, 1, self.timestamp)
                    self.insert_unlabeled += 1

        # out of the nearest microcluster
        if out:
            if label != -1:
                self.insert_directly(point, label, 1)
                self.create_labeled += 1
                self.counter[label] += 1
            elif re > self.re_threshold:
                self.insert_directly(point, label_pred, re)
                self.create_reliable += 1
                self.counter[label_pred] += 1
            else:
                self.insert_directly(point, -1, 1, labeled=False)
                self.create_unlabeled += 1

    @staticmethod
    def _distance(point_a, point_b):
        return math.sqrt(utils.math.minkowski_distance(point_a, point_b, 2))

    def _find_fixed_radius_nn(self, x):
        min_distance = math.inf
        nearest_mc = None
        fixed_radius_nn = []
        for i, mc in enumerate(self.labeled_micro_clusters):
            distance = self._distance(mc.calc_center(self.timestamp), x)
            if distance < self.clustering_threshold:
                fixed_radius_nn.append((mc, distance))
            if distance < min_distance:
                min_distance = distance
                nearest_mc = mc
        sorted_fixed_radius_nn = sorted(fixed_radius_nn, key=operator.itemgetter(-1))
        sorted_fixed_radius_nn.append((nearest_mc, min_distance))
        return fixed_radius_nn

    def _get_k_nearest_clusters(self, point, clusters, n_neighbors: int = 1):
        points = ((cluster, self._distance(cluster.calc_center(self.timestamp), point)) for i, cluster in
                  enumerate(clusters))
        return sorted(points, key=operator.itemgetter(-1))[:n_neighbors]

    def _get_closest_cluster_key_dis(self, point, clusters):
        min_distance = math.inf
        key = -1
        for k, cluster in enumerate(clusters):
            center = cluster.calc_center(self.timestamp)
            distance = self._distance(center, point)
            if distance < min_distance:
                min_distance = distance
                key = k
        return key, min_distance

    def _merge_two_closest_micro_cluster(self, unlabeled=False):
        # Merge the two closest micro-clusters
        closest_a = 0
        closest_b = 0
        min_distance = math.inf
        if unlabeled:
            micro_clusters = self.unlabeled_micro_clusters
        else:
            micro_clusters = self.labeled_micro_clusters

        for i, mc_a in enumerate(micro_clusters):
            for j, mc_b in enumerate(micro_clusters):
                if i <= j:
                    continue
                dist = self._distance(mc_a.calc_center(self.timestamp), mc_b.calc_center(self.timestamp))
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j
        self.labeled_micro_clusters[closest_a].merge(self.labeled_micro_clusters[closest_b])
        self.labeled_micro_clusters.pop(closest_b)

    class BufferItem:
        def __init__(self, x, y, timestamp, covered) -> None:
            self.x = x
            self.y = y
            self.timestamp = (timestamp,)
            self.covered = covered


class ReMicroCluster(metaclass=ABCMeta):
    r"""
        weight: 通过衰减计算后的样本数 N
    """

    def __init__(self, x, label, timestamp, decaying_factor, re=1) -> None:
        self.label = label
        self.last_edit_time = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor

        self.N = 1
        self.re = re
        self.linear_sum = x
        self.squared_sum = {i: (x_val * x_val) for i, x_val in x.items()}

    def calc_norm_cf1_cf2(self,):
        # |CF1| and |CF2| in the paper
        sum_of_squares_cf1 = 0
        sum_of_squares_cf2 = 0
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            sum_of_squares_cf1 += val_ls * val_ls
            sum_of_squares_cf2 += val_ss * val_ss
        # return |CF1| and |CF2|
        return math.sqrt(sum_of_squares_cf1), math.sqrt(sum_of_squares_cf2)

    def calc_weight(self, timestamp):
        return self._weight(self.fading_function(timestamp - self.last_edit_time))

    def _weight(self, fading_factor):
        return self.N * fading_factor

    def calc_center(self, timestamp):
        center = {key: val / self.N for key, val in self.linear_sum.items()}
        return center

    def calc_radius(self, timestamp):
        norm_cf1, norm_cf2 = self.calc_norm_cf1_cf2()
        diff = (norm_cf2 / self.N) - ((norm_cf1 / self.N) ** 2)
        radius = math.sqrt(diff) if diff > 0 else 0
        return radius

    def insert(self, x, label, re, timestamp):
        self.N += 1
        self.label = label
        self.re = re
        self.last_edit_time = timestamp
        for key, val in x.items():
            try:
                self.linear_sum[key] += val
                self.squared_sum[key] += val * val
            except KeyError:
                self.linear_sum[key] = val
                self.squared_sum[key] = val * val

    def replace(self, x, label, re, timestamp):
        self.N = 1
        self.label = label
        self.re = re
        self.creation_time = timestamp
        self.last_edit_time = timestamp
        for key, val in x.items():
            try:
                self.linear_sum[key] = val
                self.squared_sum[key] = val * val
            except KeyError:
                self.linear_sum[key] = val
                self.squared_sum[key] = val * val

    def merge(self, cluster):
        self.N += cluster.N
        for key in cluster.linear_sum.keys():
            try:
                self.linear_sum[key] += cluster.linear_sum[key]
                self.squared_sum[key] += cluster.squared_sum[key]
            except KeyError:
                self.linear_sum[key] = cluster.linear_sum[key]
                self.squared_sum[key] = cluster.squared_sum[key]
        if self.last_edit_time < cluster.creation_time:
            self.last_edit_time = cluster.creation_time

    def fading_function(self, time, epsilon=1):
        return 2 ** (-self.decaying_factor * time * epsilon)

    def update(self, timestamp, epsilon):
        ff = self.fading_function(timestamp - self.last_edit_time, epsilon=epsilon)
        self.re *= ff
        # self.re = 1 if random.random() < 0.1 else self.re
        return self.re
