<div align="center">

# ON THE CLOUD DETECTION FROM BACKSCATTERED IMAGES GENERATED FROM A LIDAR-BASED CEILOMETER: CURRENT STATE AND OPPORTUNITIES
Alessio Barbaro Chisari, Alessandro Ortis, Luca Guarnera, Wladimiro Carlo Patatu, Rosaria Ausilia Giandolfo, Emanuele Spampinato, Sebastiano Battiato, Mario Valerio Giuffrida

[![Conference](https://img.shields.io/badge/ICIP-2024)]

</div>

# Overview

This repository contains the code to run and reproduce the experiments for federating the ROCKET algorithm in the following settings:
- Centralized (non-federated)
- Federated baseline (FedAvg with all labeled data)
- Semi-Supervised Federated Learning with Image Reconstruction

Please cite as:

```bibtex
@inproceedings{casella2024fedrec,
  author  = {Casella, Bruno and Chisari, Alessio Barbaro and Aldinucci, Marco and Battiato, Sebastiano and Giuffrida, Mario Valerio},
  title   = {Federated Learning in a Semi-Supervised Environment for Earth Observation Data,
  booktitle    = {32nd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, {ESANN} 2024, Bruges, Belgium, October 9-11, 2024},
  year         = {2024},
  doi          = {},
  pages = {},
  publisher = {},
  url = {}
}
```

# Abstract
> We propose FedRec, a federated learning workflow taking advantage of unlabelled data in a semi-supervised environment to assist in the training of a supervised aggregated model. In our proposed method, an encoder architecture extracting features from unlabelled data is aggregated with the feature extractor of a classification model via weight averaging. The fully connected layers of the supervised models are also averaged in a federated fashion. We show the effectiveness of our approach by comparing it with the state-of-the-art federated algorithm, an isolated and a centralised baseline, on novel cloud detection datasets.


## Usage
- Clone this repo: `git clone`
- Install the requirements: `pip install -r requirements.txt`
- Download and unzip a sample of the dataset (50 image per each class of each dataset): `chmod a+x download_data.sh` and `./download_data.sh`
- Run the experiments: `python3 code/federated.py`
- Run `python3 code/federated.py --help` to check all the available training options. Use `--debug` to disable WandB metrics tracking. Note that right now there may be a bug that still asks for your WandB project and entities. If so, remove all the parts of the code containing WandB stuff, or simply create an account on WandB and adjust the project and entity according to your credentials.

## Example
- Centralized: `python3 federated.py --debug -e 100 -r 1`
- Federated baseline: `python3 federated.py --debug`
- FL pipeline with image denoising: `python3 federated.py --debug --denoising`

## Results
The results directory reports accuracies and F1-scores of all the considered approaches (proposed method and baselines).

## Contributors
* Bruno Casella <bruno.casella@unito.it>

* Alessio Barbaro Chisari <alessio.chisari@phd.unict.it>

* Mario Valerio Giuffrida <valerio.giuffrida@nottingham.ac.uk>
