## Code Repository for "A Pooled Cell Painting CRISPR Screening Platform Enables de novo Inference of Gene Function by Self-supervised Deep Learning"
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-models-yellow.svg)](https://huggingface.co/insitro/cp-posh)

## Installation

Create environment, install dependencies and activate the environment

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install

# activating the environment for running shell scripts
pixi shell

# for running jupyter notebooks
pixi run jupyter lab
```

## Dataset Availability

The datasets are publicly available in Amazon S3 at the following URLs:

- [124 gene CRISPR KO Morphology dataset](s3://insitro-research-2023-cellpaint-posh/single_cell_tile_images/morphology_124)
- [300 gene CRISPR KO Mechanism of Action (MoA) dataset](s3://insitro-research-2023-cellpaint-posh/single_cell_tile_images/moa_300/)
- [1640 gene CRISPR KO Druggable Genome dataset](s3://insitro-research-2023-cellpaint-posh/single_cell_tile_images/druggable_genome_1640/)


## POSH Barcode Sequencing and Assignment to CellPainting images

Tutorial for using the POSH barcode sequencing data to assign barcodes to CellPainting images can be found [here](notebooks/1_sequence_and_assign_barcodes_to_cellpaint_images.ipynb)

## CP-DINO model

Scripts for Training CP-DINO model can be found [here](scripts/training/)

Scripts for Inference using CP-DINO model can be found [here](scripts/inference/)

### Analysis of CP-DINO Embeddings

Notebooks for analysis and evaluation of CP-DINO embeddings are available for the following datasets:

- **Morphology 124 Dataset**
    - [Analysis notebook](notebooks/2_analysis_morphology_124.ipynb)

- **MoA 300 Dataset**
    - [Analysis notebook](notebooks/3_analysis_cpdino_300.ipynb)

- **Druggable Genome 1640 Dataset**
    - [Analysis notebook](notebooks/analysis/4_cpdino_1640_analysis.ipynb)

### Training

Command for training CP-DINO model on 124 gene CRISPR KO Morphology dataset

Note: User may need to enable access to .sh scripts via `chmod 700 ./scripts/training/cpdino*`

```bash
./scripts/training/cpdino_300_fp32_100ep.sh
./scripts/training/cpdino_1640_fp32_100ep.sh
./scripts/training/cpdino_1640_with_pS6_fp32_100ep.sh
```

### Pretrained Model Weights

Pretrained model weights for CP-DINO can be found [here](https://huggingface.co/insitro/cp-posh).

### Inference

Command for inference using CP-DINO model on 124 gene CRISPR KO Morphology dataset

```bash
./scripts/inference/cpposh_124_inference.sh
./scripts/inference/cpposh_300_inference.sh
./scripts/inference/cpposh_1640_inference.sh
```
