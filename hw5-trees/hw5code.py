import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    
    feature_vector = np.array(feature_vector) # на всякий случай, вдруг передадут pd.Series
    target_vector = np.array(target_vector) # на всякий случай, вдруг передадут pd.Series
    
    if feature_vector is None or feature_vector.size == 0:
        return None, None, None, None

    R = len(feature_vector)
    
    # сначала всё посортируем: индексы, вектор признаков и вектор таргетов
    idxs = np.argsort(feature_vector)
    sorted_features = feature_vector[idxs]
    sorted_targets = target_vector[idxs]
    
    # посчитаем уникальные значения признаков и сделаем из них пороги
    unique_features, cnts = np.unique(sorted_features, return_counts=True)
#     print(cnts)
    thresholds = (unique_features[1:] + unique_features[:-1]) / 2 # считаем для соседей и крайние элементы обрезаем
#     print(thresholds)
    # положительные объекты посчитаем через cumsum для каждого сплита, все объекты -- просто длина текущего поддерева
    size = np.arange(1, len(sorted_targets))
    l_pos = np.cumsum(sorted_targets[:-1]) / size
    r_pos = np.cumsum(sorted_targets[-1:0:-1]) / size
    r_pos = np.array(list(reversed(r_pos)))
    
    H_L = 1 - l_pos ** 2 - (1 - l_pos) ** 2
    H_R = 1 - r_pos ** 2 - (1 - r_pos) ** 2
    
#     $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
#     $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
#     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
    
    ginis = (- size * H_L - np.arange(len(sorted_targets) - 1, 0, -1) * H_R) / R
    ginis = ginis[(np.cumsum(cnts) - 1)[:-1]] # тк у нас признаки могут быть неуникальные, смотрим именно на количество вхождений
    
    if ginis is None or ginis.size == 0:
        return None, None, None, None
    # -1 потому что надо получить индексы, а индексы с нуля
    idx = np.argmax(ginis)
#     print(len(thresholds), len(ginis))
    return thresholds, ginis, thresholds[idx], ginis[idx]
     


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
#         print(self._feature_types)

    def _fit_node(self, sub_X, sub_y, node):
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        if np.all(sub_y == sub_y[0]): # критерий останова: все объекты принадлежат одному классу (было != вместо ==)
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]): # признаки надо перебирать с нуля
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count # было наоборот
                # сортируем по ratio, возвращаем признаки
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                # array от map не строится, надо list
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if feature_vector is None or feature_vector.size == 0: # если вектор пустой, а не длины 3
                continue
            # если уникальный признак 1, тоже не будем разбивать (без этого у меня падает взятие argmax для ginis)
            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": # с большой буквы было
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError
                    
        # критерий останова -- ни по одному признаку нельзя разбить выборку
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # первый список, первая пара 
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"]) # для y тоже нужен logical not

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        x = np.array(x)
        if node['type'] == 'terminal':
            return node['class']
        # рекурсивно спускаемся до терминальной вершины
        feature = node['feature_split']
        left = node['left_child']
        right = node['right_child']
        if self._feature_types[feature] == 'categorical':
            if x[feature] in node['categories_split']:
                return self._predict_node(x, left) 
            return self._predict_node(x, right)
        elif self._feature_types[feature] == 'real':
            if x[feature] < node['threshold']:
                return self._predict_node(x, left)
            return self._predict_node(x, right)
        return None

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        X = np.array(X)
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted).astype(int)
    
    
    
    def get_params(self, deep):
        return dict(feature_types=self._feature_types)
