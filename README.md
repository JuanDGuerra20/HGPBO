# Bidirectional Information FLow (BIF) - A Sample Efficient Hierarchical Gaussian Process for Bayesian Optimization

## Abstract
Hierarchical Gaussian Process (GP) models for Bayesian optimization decompose complex tasks into subtasks, but their information flow is unidirectional. This one-way coupling limits sample efficiency and slows convergence. We propose Bidirectional Information Flow (BIF), an efficient hierarchical GP framework that establishes bidirectional information exchange between parent and child GPs. BIF retains the modular structure of hierarchical models—each child GP focuses on a subtask—while introducing top-down feedback to continually refine child models during online learning. This mutual exchange improves sample efficiency, enables robust training even with mostly simulated subtask data, and allows modular reuse of learned subtask models. BIF outperforms conventional hierarchical GP-BO methods, achieving up to $85\%$ and $>100\%$ higher $R^2$ scores for the parent and child respectively, on synthetic and real-world neurostimulation optimization tasks.
## Guidelines

This repository is the original work of the authors of the above paper. To use this repository, we provide a requirements file.
Each test is in a python file that simply needs to be run. For synthetic 2D experiments, run efficient_general_2D.py in 
the model_testing folder with dataset=6. For 3D testing, navigate to the efficient_general.py file in 3D_testing
folder with dataset_num=6. For the neural dataset tests, run general_neural.py in the home branch.

For the additional experiments of modularity, nonlinearity, and noise, the corresponding files are aptly named 
with the exception of the noise experiments being in the test.py file. The correct folder structure is required with folders
outlining the experiment type and model type in the form of model_testing\modularity_2D\lossless_efficient.

For more information please contact the appropriate author in the publication.
 
