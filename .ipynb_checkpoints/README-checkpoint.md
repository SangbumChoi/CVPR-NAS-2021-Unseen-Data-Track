# Introduction
Thanks for taking part in our competition and downloading the starting kit! We've compiled this readme to show
you how to run all the scripts that we'll use to evaluate your submissions, as well as give you a quick tour of what
we've included. Please feel free to reach out to use for help or clarifications, using the contact details on the
competition page. Additionally, check out the CodaLab wiki here: https://github.com/codalab/codalab-competitions/wiki

# Competition Aim
We'd like you design a NAS algorithm that can generate excellent models for arbitrary datasets. This will take the form
of a class NAS inside a file named nas.py, where the NAS class contains a 'search' method. See our sample submission 
to see exactly what this will need to look like. We're going to pass a bunch of novel datasets to your NAS class, and 
you will be scored against the benchmark set by our sample submission. Good luck!

## Walkthrough
Check out the Jupyter Notebook `Walkthrough.ipynb` to see a submission and the entire evaluation
pipeline in action.

## Features of A Valid Submission
 * `nas.py`, containing
    * `NAS` class
        * `NAS.search(train_x, train_y, valid_x, valid_y, metadata)` -> returns `model`
 * other libraries or helpers: anything imported by `nas.py`  (optional)   
* `metadata` file: 
    * `description`: Make sure this field is there, but its contents doesn't matter
    * 'full_training': `true` to run the full evaluation procedure ,`false` to run only a few batches of 
    data during training. This latter option is useful to test that your model is compatible with the evaluation programs. 
Bundle all the above into a zip file and upload it to the CodaLab competition page:
* `submission.zip    `
    * `nas.py`
    * other imports  (optional)
    * `metadata`

See our sample submission for more information, and please reach out if you have any questions.

# Competition Framework
### Setup
The compute workers have CUDA 10.1 installed, with torch==1.8.0 and torchvision==0.9.0
If there are other packages that you'd like that aren't installed on the workers, let us know and we can have a 
discussion about installing them.


### Data
We've included some tiny datasets in the starting kit. Download the public data to see the exact same datasets that
the development phase of the competition will use. While the public data has the ground-truth test values included,
these will be hidden from your scripts when you submit them to our servers, so don't rely on them! Each dataset is a 
4D tensor of shape (n_datapoints, channels, h, w). Each task in the competition is a n-class classification problem,
where each class is evenly balanced in each of the train, validation, and test data splits. Included with each dataset
is a metadata file, that tells you information about the classification task and how models for this dataset will be
trained. The schema of this metadata file is further detailed in our sample submission

### Evaluation Pipeline
Your NAS class will be evaluated in two steps; ingestion and then scoring. 

#### Ingestion:
The ingestion program will, for each dataset:
    
1) Initialize your NAS class
2) Pass the train and validation data to NAS.search()
3) Re-initialize the found model to reset any found weights 
4) Train the found model
5) Save the test predictions from the best validation epoch

#### Scoring:
The scoring program will, for each dataset:

1) Load the ground truth test values
2) Load the saved test predictions from your model
3) Compute the prediction accuracy
4) Adjust score based on our benchmark: `score = 10 * (test_accuracy - benchmark_accuracy)/(100 - benchmark_accuracy)`
   There are 10 possible points to score per each dataset. 
5) Add per-dataset score to overall score

Both these scripts are included in the starting kit, so take a look to see exactly how it all works. 


# Test your submission locally:
Use the included makefile to run your submission through the ingestion and scoring programs in the same way that we
will:

* `make ingest data=$data_directory`
* `make score data=$data_directory`

or 
    `make all data=$data_directory` to do both.

`$data_directory` should point to the top level data directory (the one that contains the directories `dataset_0`, 
`dataset_1`, etc.) that you want to run the scripts over. For example:
    
`make all data=sample_data` or `make all data=public_data`

# Submit
Upload your submission zip to the CodaLab competition page. It will take a few hours per dataset to train each model,
so take that into account in addition to the runtime of your NAS algorithm while waiting for results. Once the submission
has been evaluated, you can see the stdout and stderr from both of the ingestion program and scoring program from the competition
site. If the submission is successful, the "ingestion_output_log" will show detailed, per-epoch training results, both in 
human-readable format and as a Python dictionary. 

**Make sure that the full_train flag in your submission metadata is set to `true`**

# Contact
Please don't hesitate to reach out to us at `cvpr-2021-nas@newcastle.ac.uk` or `rob@geada.net` if you have any questions! 
    
# License
All code and datasets (excluding development_dataset_1) distributed through this competition were created by 
the organizers of the NASComp-2021 Competition, and are licensed through the GPL v3 license. 

