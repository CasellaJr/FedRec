# Image-Reconstruction-as-an-Effective-Way-for-Optimizing-Federated-Learning

> ABSTRACT

BIBTEX:
```bibtex
@inproceedings{,
  author  = {},
  title   = {},
  booktitle = {},
  doi = {},
  location = {},
  month = ,
  url = {},
  year = {},
  volume  = {},
  number  = {},
  pages   = {X}
}
```

This repository contains the code to run and reproduce the experiments for federating the ROCKET algorithm in the following settings:
- Centralized (non-federated)
- Federated baseline (FedAvg with all labeled data)
- Semi-Supervised Federated Learning with Image Reconstruction

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
