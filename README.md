# learning-to-smell-starter-kit
![AIcrowd-Logo](https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png)

Starter kit for getting started in the [AIcrowd Learning 2 Smell Challenge](https://www.aicrowd.com/challenges/learning-to-smell).

# Installation

```
git clone https://github.com/AIcrowd/learning-to-smell-starter-kit
cd learning-to-smell-starter-kit
pip install -r requirements.txt
```

# Data download
Download all the files from the [AIcrowd Resources page](https://www.aicrowd.com/challenges/learning-to-smell/dataset_files),
and put them in the `data/` folder. This should give you a folder structure similar to : 

```
.
├── data
│   ├── test.csv
│   ├── train.csv
│   └── vocabulary.txt
```


**NOTE**: If you have not accepted the challenge rules (by clicking on the `Participate` button), you will be asked to agree to the Rules of the competition at this point.

# Basic Usage

## Generate Random Predictions
```
python random_predict.py 
```

This should generate a submission.csv file, which you can upload by clicking on `Create Submission` on the challenge page.

## Predictions by Graph Neural Networks
**Coming Soon**

## Predictions by using Molecular Fingerprints
**Coming Soon**

# Author
S.P. Mohanty <mohanty@aicrowd.com>