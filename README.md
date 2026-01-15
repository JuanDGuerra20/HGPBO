# Bidirectional Information FLow (BIF) - A Sample Efficient Hierarchical Gaussian Process for Bayesian Optimization

## Abstract
Hierarchical Gaussian Process (H-GP) models divide problems into different subtasks, allowing for different models to address each part, making them well-suited for problems with inherent compositional structure. However, existing H-GP frameworks typically employ one-way information sharing — either top-down or bottom-up — which limits sample efficiency and slows convergence. We propose Bidirectional Information Flow (BIF), an efficient framework that defines a hierarchy of probabilistic models, where children and parent each represent beliefs over functions at different levels of aggregation for online training. BIF retains the modular structure of hierarchical models — the parent conditions its own posterior on child summaries, treating them as structured priors — while introducing top-down feedback to softly decompose environment observations from the parent into sup-responses using the children's current predictive beliefs. This mutual exchange improves sample efficiency, enables robust training, and allows modular reuse of learned subtask models. We prove analytically the regret of a GP with a learned kernel scales linearly with the mismatch to the true kernel and is upper bounded by the mismatch of the children in hierarchical cases. BIF outperforms conventional H-GP Bayesian Optimization methods, achieving up to 4$\times$  higher $R^2$ scores for the parent, on synthetic and real-world neurostimulation optimization tasks.

## Guidelines

This repository is the original work of the authors of the above paper. To use this repository, we provide a requirements file.
Each test is in a python file that simply needs to be run. For synthetic 2D experiments, run efficient_general_2D.py in 
the model_testing folder with dataset=3. For 3D testing, navigate to the efficient_general.py file in 3D_testing
folder with dataset_num=6. For the neural dataset tests, run general_neural.py in the home branch.

For the additional experiments of modularity, nonlinearity, and noise, the corresponding files are aptly named 
with the exception of the noise experiments being in the test.py file. The correct folder structure is required with folders
outlining the experiment type and model type in the form of model_testing\modularity_2D\lossless_efficient.

For more information please contact the appropriate author in the publication.
 
