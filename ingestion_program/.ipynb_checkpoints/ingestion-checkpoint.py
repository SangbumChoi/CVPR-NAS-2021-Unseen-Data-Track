import os
import sys
import numpy as np

from nascomp.helpers import get_dataset_paths, load_datasets
from nascomp.torch_evaluator import torch_evaluator


if __name__ == '__main__':
    # this is the location of the input data
    input_dir = sys.argv[2]

    # this is where we'll write the models' predictions over the test dataset
    output_dir = sys.argv[3]

    # this is the location of the competitor's code
    submission_dir = sys.argv[6]
    with open(os.path.join(submission_dir, "metadata"), "r") as f:
        metadata = f.read().split("\n")
    full_train = True
    for line in metadata:
        if 'full_training' in line and 'false' in line:
            full_train = False

    # import competitor codebase
    sys.path.append(submission_dir)
    import nas

    # iterate through input datasets
    full_results = {}
    for dataset_path in get_dataset_paths(input_dir):
        (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_datasets(dataset_path)

        # initialize user algorithm
        nas_algorithm = nas.NAS()

        # run NAS over this dataset
        print("===== RUNNING NAS FOR {} =====".format(metadata['name']))
        model = nas_algorithm.search(train_x, train_y, valid_x, valid_y, metadata)

        # package data for evaluator
        data = (train_x, train_y), (valid_x, valid_y), test_x

        # train model for $n_epochs, recover test predictions from best validation epoch
        results = torch_evaluator(model, data, metadata, n_epochs=64, full_train=full_train)
        predictions = results['test_predictions']
        train_details = {k: v for k, v in results.items() if k!='test_predictions'}
        full_results[metadata['name']] = train_details

        # save these predictions to the output dir
        np.save(os.path.join(output_dir, "predictions_{}.npy".format(metadata['name'])), predictions)

    print("=== FINISHED EVALUATION ===")
    print(full_results)

