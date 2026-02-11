# Physically Consistent Evaluation of Commonly Used Near-Field Models

We provide in this repository the code and the model parameters via the SENF (Sampled Electromagnetic Near Field) dataset for our sampled near-field REMS model.

## How to run the code
- Download the SENF dataset [here](https://iis-people.ee.ethz.ch/~datasets/senf/) and place it in the senf folder 
- Install [uv](https://docs.astral.sh/uv/)
- Run `uv run main.py`

## How to create the SENF
We also provide the Ansys HFSS project `evaluation_of_near_field_models.aedt` and the python scripts (in the directory `export_scripts`) to create the dataset on your own.

If you are using the code or the Ansys HFSS project (or parts of it) for a publication, then you must cite the following paper:

Physically Consistent Evaluation of Commonly Used Near-Field Models

(c) 2026 Georg Schwan, Alexander Stutz-Tirri, and Christoph Studer

e-mail: gschwan@ethz.ch, alstutz@ethz.ch, studer@ethz.ch

Version history

Version 0.1: gschwan@ethz.ch - initial version for GitHub release

