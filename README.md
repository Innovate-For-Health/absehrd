# Automated Brewing of Synthetic Electronic Health Record Data (ABSEHRD)

## Background
A consolidated software package for producing and validating synthetic electronic health record data.

The currently incorporated synthetic data generator is CorGAN (Torfi et al. 2020. CorGAN: Correlation-Capturing Convolutional Generative Adversarial Networks
for Generating Synthetic Healthcare Records). 

## Installation

Download the zip file from GitHub or clone the repository locally:

```
git clone https://github.com/Innovate-For-Health/absehrd
```

Navigate to the directory and create a conda environment:

```
conda env create -f absehrd.yml
conda activate absehrd
```

## Usage

### Command line interface
To see an overview of the command line interface (CLI) usage, run
```
python absehrd.py -h
```

The CLI has four main executable tasks labeled 'train', 'generate', 'realism', and 'privacy'.
You can view all arguments for each action, by specifying the task name and the
help flag as for the 'train' task below:
```
python absehrd.py train -h
```

To test out the package, create a sample dataset:
```
python example_dataset.py
```

Train a synthetic data generator on the example dataset:
```
python absehrd.py train --file_data examples/example_train.csv \
                        --outprefix_train examples/example_model \
                        --verbose
```

Generate synthetic samples from the trained generator:
```
python absehrd.py generate --file_model examples/example_model.pkl \
                           --outprefix_generate examples/example_synthetic \
                           --generate_size 5000
```

Validate the realism of the synthetic dataset:
```
python absehrd.py realism --outprefix_realism examples/example_realism \
                          --file_realism_real_train examples/example_train.csv \
                          --file_realism_real_test examples/example_test.csv \
                          --file_realism_synth examples/example_synthetic.csv \
                          --outcome binary01 \
                          --analysis_realism gan_train_test \
                          --output_realism summary 
```

Validate the privacy-preserving properties of the synthetic dataset:
```
python absehrd.py privacy --outprefix_privacy examples/example_realism \
                          --file_privacy_real_train examples/example_train.csv \
                          --file_privacy_real_test examples/example_test.csv \
                          --file_privacy_synth examples/example_synthetic.csv \
                          --analysis_privacy nearest_neighbors \
                          --output_privacy summary
```

And when finished remember to deactivate the environment:
```
conda deactivate
```

### Package interface
See absehrd.ipynb for example usage and additional documentation.

## Acknowledgments
Thanks to ajgokhale and cynthiakwu for contributions 
to this project. This project was funded by Johnson & Johnson through the 
Innovate for Health Data Science Fellowship Program (https://innovateforhealth.berkeley.edu/), 
a collaboration between UC Berkeley, UCSF, and Johnson & Johnson.

