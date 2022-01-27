# create conda environment
conda env create -f absehrd.yml
conda activate absehrd

# To see an overview of the command line interface (CLI) usage, run
python absehrd.py -h

# The CLI has four main executable tasks labeled 'train', 'generate', 'realism', and 'privacy'.
# You can view all arguments for each action, by specifying the task name and the
# help flag as for the 'train' task below:
python absehrd.py train -h

# To test out the package, create a sample dataset:
python example_dataset.py

# Train a synthetic data generator on the example dataset:
python absehrd.py train --file_data examples/example_train.csv \
                        --outprefix_train examples/example_model \
                        --verbose


# Generate synthetic samples from the trained generator:
python absehrd.py generate -h
python absehrd.py generate --file_model examples/example_model.pkl \
                           --outprefix_generate examples/example_synthetic \
                           --generate_size 5000


# Validate the realism of the synthetic dataset:
python absehrd.py realism -h
python absehrd.py realism --outprefix_realism examples/example_realism \
                          --file_realism_real_train examples/example_train.csv \
                          --file_realism_real_test examples/example_test.csv \
                          --file_realism_synth examples/example_synthetic.csv \
                          --outcome binary01 \
                          --analysis_realism gan_train_test \
                          --output_realism summary 


# Validate the privacy-preserving properties of the synthetic dataset:
python absehrd.py privacy -h
python absehrd.py privacy --outprefix_privacy examples/example_realism \
                          --file_privacy_real_train examples/example_train.csv \
                          --file_privacy_real_test examples/example_test.csv \
                          --file_privacy_synth examples/example_synthetic.csv \
                          --analysis_privacy nearest_neighbors \
                          --output_privacy summary


# And when finished remember to deactivate the environment:
conda deactivate
