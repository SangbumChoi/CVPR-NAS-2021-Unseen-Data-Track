import json
import os
import sys
import numpy as np


if __name__ == '__main__':
    # where model predictions and test_y are saved
    input_dir = sys.argv[1]

    # the location of test_y
    reference_dir = os.path.join(input_dir, 'ref')

    # the location of model predictions
    prediction_dir = os.path.join(input_dir, 'res')

    # where to save scores.txt
    output_dir = sys.argv[2]

    overall_score = 0
    out = []

    # iterate through the reference datasets
    paths = sorted([os.path.join(reference_dir, d) for d in os.listdir(reference_dir) if 'dataset' in d])
    for path in paths:

        # load the reference values
        ref_y = np.load(os.path.join(path, 'test_y.npy'))

        # load the dataset_metadata for this dataset
        with open(os.path.join(path, 'dataset_metadata'), "r") as f:
            metadata = json.load(f)

        print("=== Scoring {} ===".format(metadata['name']))
        index = metadata['name'][-1]

        # load the model predictions
        pred_y = np.load(os.path.join(prediction_dir, "predictions_{}.npy".format(metadata['name'])))

        # compute accuracy
        score = sum(ref_y == pred_y)/float(len(ref_y)) * 100
        print("  Raw score:", score)
        print("  Benchmark:", metadata['benchmark'])

        # adjust score according to benchmark
        point_weighting = 10/(100 - metadata['benchmark'])
        score -= metadata['benchmark']
        score *= point_weighting

        print(" Adjusted:  ", score)

        # add per-dataset score to overall
        overall_score += score

        # add to scoring stringg
        out.append("Dataset_{}_Score: {:.3f}".format(index, score))
    out.append("Overall_Score: {:.3f}".format(overall_score))

    # save score for leaderboard consideration
    print(out)
    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        f.write("\n".join(out))





