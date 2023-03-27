# SOMperf: SOM performance metrics and quality indices

**This package is in its early phase of development. SOM performance metrics have all been tested pretty well, but they may still contain bugs. Please report them in an issue if you find some.**

If you found this library useful in your work, please cite following preprint:

> Forest, Florent, Mustapha Lebbah, Hanane Azzag, and Jérôme Lacaille (2020). A Survey and Implementation of Performance Metrics for Self-Organized Maps. arXiv, November 11, 2020. https://doi.org/10.48550/arXiv.2011.05847.

## Installation

This module was written for Python 3 and depends on following libraries:

* numpy
* pandas
* scipy
* scikit-learn

SOMperf can be installed easily using the setup script:

```shell
python3 setup.py install
```

It might be available in PyPI in the future.

## Getting started

SOMperf contains 2 modules: `metrics`, containing all internal and external quality indices, and `utils`, containing utility functions for SOMs (distance and neighborhood functions).

Metric functions usually take several of following arguments:

* `som`: a self-organizing map model with _K_ prototypes/code vectors in dimension _D_ given as a _K X D_-numpy array
* `x`: data matrix with _N_ samples in dimension _D_, given as a _N X D_-numpy array
* `d`: a pre-computed pairwise (non-squared) euclidean distance matrix between samples and prototypes, given as a _N X K_-numpy array
* `dist_fun`: a function computing the distance between two units on the map, such that `dist_fun(k, l) == 1` iff `k` and `l`are neighbors. Distance function on usual grid topologies are available in `somperf.utils.topology`.
* `neighborhood_fun`: neighborhood kernel function used in the SOM distortion loss. Usual neighborhood functions are available in `somperf.utils.neighborhood`.

Neighborhood preservation and Trustworthiness also take an additional `k` argument for the number of neighbors to consider.

Here is a quick example using minisom to compute metrics on an 8-color dataset and a 10-by-10 map:

```python
import numpy as np
from minisom import MiniSom

from somperf.metrics import *
from somperf.utils.topology import rectangular_topology_dist

# 8 colors
X = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 1.0, 1.0],
              [0.5, 0.5, 0.5],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 1.0],
              [1.0, 0.0, 1.0]])

# define and train 10-by-10 map
map_size = (10, 10)
som = MiniSom(map_size[0], map_size[1], X.shape[-1], sigma=1.0, learning_rate=1.0, random_seed=42)
som.random_weights_init(X)
som.train_random(X, 10000)

# get weights as a (100, 3) array
weights = som.get_weights().reshape(map_size[0]*map_size[1], -1)

# compute a few metrics
print('Topographic product = ', topographic_product(rectangular_topology_dist(map_size), weights))
print('Neighborhood preservation = ', neighborhood_preservation(1, weights, X))
print('Trustworthiness = ', trustworthiness(1, weights, X))
```

Here are the results:

```python
0.3002313673993011  # TP > 0 is no surprise, because a (10, 10) map is too large for our 8-color dataset
0.9375  # original neighbors are not always assigned to neighboring prototypes
1.0  # perfect trustworthiness means that any neighboring prototypes correspond to original neighboring samples
```

Label-based metrics, also called external indices, rather take as inputs the cluster labels `y_pred` and the ground-truth class labels `y_true`, except the Class scatter index that also depends on the map topology (`dist_fun`).

## List of metrics

### Internal

* [x] Combined error [5]
* [x] Distortion (SOM loss function) [4,8,11]
* [x] Kruskal-Shepard error [3,7]
* [x] Neighborhood preservation [10]
* [x] Quantization error
* [x] Topographic error
* [x] Topographic product [1,2]
* [x] Trustworthiness [10]
* [x] Silhouette :arrow_right: `sklearn.metrics.silhouette_score`
* [x] Davies-Bouldin :arrow_right: `sklearn.metrics.davies_bouldin_score`
* [x] Topographic function [12]
* [x] C Measure [13]

### External (label-based)

* [x] Adjusted Rand index (ARI) :arrow_right: `sklearn.metrics.adjusted_rand_score`
* [x] Class scatter index [3]
* [x] Completeness :arrow_right: `sklearn.metrics.completeness_score`
* [x] Entropy [3]
* [x] Homogeneity :arrow_right: `sklearn.metrics.homogeneity_score`
* [x] Normalized Minor class occurrence [3] (= 1 - purity)
* [x] Mutual information :arrow_right: `sklearn.metrics.mutual_info_score`
* [x] Normalized mutual information (NMI) :arrow_right: `sklearn.metrics.normalized_mutual_info_score`
* [x] Purity
* [x] Unsupervised clustering accuracy 

## List of SOM utilities

### Map distance functions

* [x] Rectangular topology
* [x] Square topology
* [ ] Hexagonal topology
* [ ] Cylindrical topology
* [ ] Toroidal topology

### Neighborhood functions

* [x] Gaussian neighborhood
* [x] Constant window neighborhood
* [ ] Triangle neighborhood
* [ ] Inverse neighborhood
* [ ] Squared inverse neighborhood
* [ ] Mexican hat neighborhood
* [ ] Clipped versions (0 if d < eps)

## Tests

All metrics have been tested to check results against manually computed values, expected behavior and/or results from research papers. Tests and visualizations are available as a jupyter notebook in the `tests/` directory.

## SOM libraries

Here is a small selection of SOM algorithm implementations:

* [SOM toolbox](https://github.com/ilarinieminen/SOM-Toolbox) ![](https://img.shields.io/github/stars/ilarinieminen/SOM-Toolbox.svg?style=social) (Matlab)
* [minisom](https://github.com/JustGlowing/minisom) ![](https://img.shields.io/github/stars/JustGlowing/minisom.svg?style=social) (Python)
* [SOMPY](https://github.com/sevamoo/SOMPY) ![](https://img.shields.io/github/stars/sevamoo/SOMPY.svg?style=social) (Python)
* [tensorflow-som](https://github.com/cgorman/tensorflow-som) ![](https://img.shields.io/github/stars/cgorman/tensorflow-som.svg?style=social) (Python/TensorFlow)
* [DESOM](https://github.com/FlorentF9/DESOM) ![](https://img.shields.io/github/stars/FlorentF9/DESOM.svg?style=social) (Python/Keras)
* SOMbrero ([CRAN](https://cran.r-project.org/web/packages/SOMbrero/index.html)/[Github](https://github.com/tuxette/SOMbrero)) ![](https://img.shields.io/github/stars/tuxette/SOMbrero.svg?style=social) (R)
* [sparkml-som](https://github.com/FlorentF9/sparkml-som) ![](https://img.shields.io/github/stars/FlorentF9/sparkml-som.svg?style=social) (Scala/Spark ML)

## References

> [1] Bauer, H.-U., & Pawelzik, K. R. (1992). Quantifying the Neighborhood Preservation of Self-Organizing Feature Maps. IEEE Transactions on Neural Networks, 3(4), 570–579. https://doi.org/10.1109/72.143371

> [2] Bauer, H.-U., Pawelzik, K., & Geisel, T. (1992). A Topographic Product for the Optimization of Self-Organizing Feature Maps. Advances in Neural Information Processing Systems, 4, 1141–1147.

> [3] Elend, L., & Kramer, O. (2019). Self-Organizing Maps with Convolutional Layers. In WSOM 2019: Advances in Self-Organizing Maps, Learning Vector Quantization, Clustering and Data Visualization (Vol. 976, pp. 23–32). Springer International Publishing. https://doi.org/10.1007/978-3-030-19642-4

> [4] Erwin, E., Obermayer, K., Schulten, K. Self-Organizing Maps: Ordering, convergence properties and energy functions. Biological Cybernetics, 67(1):47-55, 1992

> [5] Kaski, S., & Lagus, K. (1996). Comparing Self-Organizing Maps. In Proceedings of International Conference on Artificial Neural Networks (ICANN).

> [6] Kohonen, T. (1990). The Self-Organizing Map. In Proceedings of the IEEE (Vol. 78, pp. 1464–1480). https://doi.org/10.1109/5.58325

> [7] Kruskal, J.B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis.

> [8] Lampinen, O. Clustering Properties of Hierarchical Self-Organizing Maps. Journal of Mathematical Imaging and Vision, 2(2-3):261–272, November 1992

> [9] Polzlbauer, G. (2004). Survey and comparison of quality measures for self-organizing maps. Proceedings of the Fifth Workshop on Data Analysis (WDA04), 67–82.

> [10] Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study. Lecture Notes in Computer Science, 2130. https://doi.org/10.1007/3-540-44668-0

> [11] Vesanto, J., Sulkava, M., & Hollmén, J. (2003). On the Decomposition of the Self-Organizing Map Distortion Measure. Proceedings of the Workshop on Self-Organizing Maps (WSOM’03), 11–16.

> [12] Villmann, T., Der, R., Martinez, T. A new quantitative measure of topology preservation in Kohonen's feature maps, Proceedings of the IEEE International Conference on Neural Networks 94, Orlando, Florida, USA, 645-648, June 1994

> [13] Goodhill, G. J., & Sejnowski, T. J. (1996). Quantifying neighbourhood preservation in topographic mappings. Proceedings of the 3rd Joint Symposium on Neural Computation, La Jolla, CA, 61–82.

## Future work

* Implement per-node metrics
* Other SOM analysis and visualization modules
