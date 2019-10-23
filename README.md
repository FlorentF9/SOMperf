# SOMperf: SOM performance metrics and quality indices

**This package is in its early phase of development. SOM performance metrics are available, but they may still contain bugs. Please report them in an issue if you find some.**

## Installation

This module was written for Python 3. It can be installed easily using the setup script:

```shell
$ python3 setup.py install
```

## Getting started

TODO

## List of metrics

### Internal

* Combined error [5] :heavy_check_mark:
* Distortion (SOM loss function) [4,8,11] :heavy_check_mark:
* Kruskal-Shepard error [3,7] :heavy_check_mark:
* Neighborhood preservation [10] :heavy_check_mark:
* Quantization error :heavy_check_mark:
* Topographic error :heavy_check_mark:
* Topographic product [1,2,12] :heavy_check_mark:
* Trustworthiness [10] :heavy_check_mark:

### External (label-based)

* Adjusted Rand index (ARI) :arrow_right: `sklearn.metrics.adjusted_rand_score`
* Class scatter index [3] :heavy_check_mark:
* Completeness :arrow_right: `sklearn.metrics.completeness_score`
* Davies-Bouldin :arrow_right: `sklearn.metrics.davies_bouldin_score`
* Entropy [3] :heavy_check_mark:
* Homogeneity :arrow_right: `sklearn.metrics.homogeneity_score`
* Normalized Minor class occurrence [3] :heavy_check_mark: (= 1 - purity)
* Mutual information :arrow_right: `sklearn.metrics.mutual_info_score`
* Normalized mutual information (NMI) :arrow_right: `sklearn.metrics.normalized_mutual_info_score`
* Purity :heavy_check_mark:
* Silhouette :arrow_right: `sklearn.metrics.silhouette_score`
* Unsupervised clustering accuracy :heavy_check_mark: 

## List of SOM utilities

### Map distance functions

* Rectangular topology :heavy_check_mark:
* Square topology :heavy_check_mark:
* Hexagonal topology :white_check_mark:
* Cylindrical topology :white_check_mark:
* Toroidal topology :white_check_mark:

### Neighborhood functions

* Gaussian neighborhood :heavy_check_mark:
* Constant window neighborhood :heavy_check_mark:
* Inverse neighborhood :white_check_mark:
* Squared inverse neighborhood :white_check_mark:
* Clipped versions (0 if d < eps) :white_check_mark:

## SOM libraries

Here is a small selection of SOM algorithm implementations:

* [SOM toolbox](https://github.com/ilarinieminen/SOM-Toolbox) (Matlab)
* [minisom](https://github.com/JustGlowing/minisom) (Python)
* [SOMPY](https://github.com/sevamoo/SOMPY) (Python)
* [tensorflow-som](https://github.com/cgorman/tensorflow-som) (Python/TensorFlow)
* [DESOM](https://github.com/FlorentF9/DESOM) (Python/Keras)
* [SOMbrero](https://cran.r-project.org/web/packages/SOMbrero/index.html) (R)
* [sparkml-som](https://github.com/FlorentF9/sparkml-som) (Scala/Spark ML)

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

## Future work

* Implement per-node metrics
* Other SOM analysis and visualization modules
