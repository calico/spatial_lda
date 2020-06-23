# Spatial-LDA

![spatial-lda](https://github.com/calico/spatial_lda/workflows/spatial-lda/badge.svg)

Spatial-LDA is a probabilistic topic model for identifying characteristic cellular microenvironments from
in-situ multiplexed imaging data such as [MIBI-ToF]() or [CODEX]().

This repository contains an implementation of the Spatial-LDA model as described in the paper
[Modeling Multiplexed Images with Spatial-LDA Reveals Novel Tissue Microenvironments](https://www.liebertpub.com/doi/full/10.1089/cmb.2019.0340).

Please cite our work if you find this tool useful.

**Modeling Multiplexed Images with Spatial-LDA Reveals Novel Tissue Microenvironments**  
Zhenghao Chen, Ilya Soifer, Hugo Hilton, Leeat Keren, and Vladimir Jojic

Journal of Computational Biology 2020.04.03; doi: http://doi.org/10.1089/cmb.2019.0340

**BibTeX**

```latex
@article{chen2020modeling,
  title={Modeling Multiplexed Images with Spatial-LDA Reveals Novel Tissue Microenvironments},
  author={Chen, Zhenghao and Soifer, Ilya and Hilton, Hugo and Keren, Leeat and Jojic, Vladimir},
  journal={Journal of Computational Biology},
  year={2020},
  publisher={Mary Ann Liebert, Inc., publishers 140 Huguenot Street, 3rd Floor New~…}
}
```

The repository also contains notebooks that generate the results and figures presented in the paper as examples of
how to use Spatial-LDA.

## Installation

`pip install spatial_lda`

## Examples

Please refer to the included notebooks below for examples of how to train a Spatial-LDA model. We include two notebooks:

### (1) Applying Spatial-LDA to a CODEX dataset of mouse spleen tissues

We apply Spatial-LDA to a dataset of mouse spleens from [Deep Profiling of Mouse Splenic Architecture with
CODEX Multiplexed Imaging](https://www.cell.com/cell/pdf/S0092-8674(18)30904-8.pdf) to validate that it recovers known spatial relationships
between immune cells in the mouse spleen.

[**Mouse Spleen Analysis**](https://drive.google.com/file/d/120835hoLuRztIZG7c0LMEewy5VI9HuZF/view?usp=sharing)

<a href="https://drive.google.com/file/d/120835hoLuRztIZG7c0LMEewy5VI9HuZF/view?usp=sharing"><img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="128"></a>

### (2) Applying Spatial-LDA to a MIBI-ToF dataset of Triple Negative Breast Cancer (TNBC) tumors

We apply Spatial-LDA to a dataset of TNBC tumors from [A Structured Tumor-Immune Microenvironment in Triple Negative
Breast Cancer Revealed by Multiplexed Ion Beam Imaging](https://www.sciencedirect.com/science/article/pii/S0092867418311000)
to identify prototypical tumor-immune microenvironments in TNBC.

[**TNBC Analysis**](https://drive.google.com/file/d/1GJpdw9jvOlR_GZpbgeBYq9NXWeeZ69aG/view?usp=sharing)

<a href="https://drive.google.com/file/d/1GJpdw9jvOlR_GZpbgeBYq9NXWeeZ69aG/view?usp=sharing"><img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="128"></a>

For convenience, we have included pre-processed versions of the data from the two datasets above under 'data/'.

## Usage

### Featurization

The Spatial-LDA model requires a dataset of index cells and neighborhood features along with an undirected graph
where nodes are index cells and edges between nodes encode index cells that should be regularized to have similar
topic priors.

We provide utilities in the `featurization` module to generate required neighborhood features
(`featurization.featurize_samples`) and adjacency matrices (`featurization.make_merged_difference_matrices`)
from dataframes containing the location and features of index and background cells.

### Training and inference

To fit a Spatial-LDA model, call `spatial_lda.model.train` on the feature matrix and difference matrix generated in
the featurization step. E.g.,

```python
spatial_lda_model = spatial_lda.model.train(train_tumor_marker_features, 
                                            train_difference_matrices, 
                                            n_topics=N_TOPICS, 
                                            difference_penalty=DIFFERENCE_PENALTY, 
                                            verbosity=1,
                                            n_parallel_processes=3,
                                            n_iters=3,
                                            admm_rho=0.1,
                                            primal_dual_mu=2)
```

To run inference - computing regularized topic weights on a pre-trained set of topics:

```python
complete_lda = spatial_lda.model.infer(
      spatial_lda_model.components_, tumor_marker_features, 
      complete_difference_matrices, difference_penalty=DIFFERENCE_PENALTY,
      n_parallel_processes=N_PARALLEL_PROCESSES)
```
