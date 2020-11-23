# Learning to Smell - Starter Kit

![AIcrowd-Logo](https://d3000t1r8yrm6n.cloudfront.net/raw_images/challenges/banner_file/525/desk_5.jpg)



- 💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/learning-to-smell
- 🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/learning-to-smell/discussion
- 🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/learning-to-smell/leaderboards

<p align="center">
 <a href="https://discord.gg/mkqquSE"><img src="https://img.shields.io/discord/657211973435392011?style=for-the-badge" alt="chat on Discord"></a>
</p>


# 💻 Installation

```
git clone https://github.com/AIcrowd/learning-to-smell-starter-kit
cd learning-to-smell-starter-kit
pip install -r requirements.txt
```

# 💾 Data download
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

# ⚙️ Basic Usage

This should generate a submission.csv file, which you can upload by clicking on `Create Submission` on the challenge page.

### Generate Random Predictions
```
python random_predict.py 
```

### Predictions by using Molecular Fingerprints

This implementation was added as Community Contribution by @latticetower (@lacemaker on AIcrowd) [here](https://discourse.aicrowd.com/t/explained-by-the-community-200-chf-cash-prize-x-5/3674/8?u=shivam).
We have ported it on official starter kit as an example.
```
python fingerprint_predict.py
```

### Predictions by Graph Neural Networks
**Coming Soon**


# 🏛 Repository Structure

We have created this sample submission repository which you can use as reference.

#### aicrowd.json
Each repository should have a aicrowd.json file with the following fields:

```
{
    "challenge_id" : "learning-to-smell",
    "grader_id": "learning-to-smell",
    "authors" : ["aicrowd-user"],
    "description" : "Learning to Smell challenge submission",
    "license" : "MIT",
    "gpu": false
}
```
This file is used to identify your submission as a part of the Learning to Smell challenge.  You must use the `challenge_id` and `grader_id` specified above in the submission. The `gpu` key in the `aicrowd.json` lets your specify if your submission requires a GPU or not. In which case, a NVIDIA-K80 will be made available to your submission when evaluation the submission.

#### Submission environment configuration
You can specify software runtime of your code by modifying the included [requirements.txt](requirements.txt). The submission support adding custom conda environment, apt packages, Dockerfile and lot's more. Guide for advanced users is present [here](https://discourse.aicrowd.com/t/how-to-specify-runtime-environment-for-your-submission/2274).

#### Debug submission

In order to test end to end pipeline, you can make debug submission as well by making `debug_submission: true` in `aicrowd.json`.

* Wouldn't be counted towards your daily submission limit, and have seperate counter
* Scores of such submission wouldn't be counted for the leaderboard
* Testing data would be same as subset of training data
* Logs of your code will be available to you

#### Code Entrypoint
The evaluator will use `run.sh` as the entrypoint, which in turn uses `predict.py`. Please remember to instantiate your class in the same or change `run.sh`.

#### Class Structure
You can refer to [random_predict.py](random_predict.py) for structuring your codes. We will be running your code as specified in `evaluator/learning_to_smell.py`. The class has placeholders for testing code, but it isn't _required_ for making submission.

```
class SuperCoolPredictor:
    """
    Below paths will be preloaded for you, you can read them as you like.
    """
    self.training_data_path = None
    self.test_data_path = None
    self.vocabulary_path = None

    """
    This function will be called for all the smiles string one by one during the evaluation.
    The return need to be a list of list. Example: [["burnt", "oily", "rose"], ["fresh", "citrus", "green"]]
    NOTE: In case you want to load your model, please do so in `predict_setup` function.
    """
    def predict(self, smile_string):
        [...]
        return SENTENCES

    """
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def predict_setup(self):
        pass
```

#### Timeouts

* Prediction setup i.e. `def predict_setup` has timeout of 10 minutes (600 seconds)
* Prediction i.e. `def predict` has timeout of 1 second per smile string.

### Local Debug

You can run your submission locally using `python3 predict.py` and verify the generated prediction csv.


# 🚀 Making submission 

To make a submission, you will have to create a private repository on [https://gitlab.aicrowd.com](https://gitlab.aicrowd.com).

You will have to add your SSH Keys to your GitLab account by following the instructions [here](https://docs.gitlab.com/ee/gitlab-basics/create-your-ssh-keys.html).
If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

*Testing that everything is set up correctly*
To test whether your SSH key was added correctly, run the following command in your terminal 

```
ssh -T git@gitlab.aicrowd.com
```

Then you can create a submission by making a *tag push* to your repository, adding the correct git remote and pushing to the remote:

```
git clone https://github.com/AIcrowd/learning-to-smell-starter-kit
cd learning-to-smell-starter-kit

# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/learning-to-smell-starter-kit.git
git push aicrowd master

# Create a tag for your submission and push (tag should start with `submission-` prefix)
git tag submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change, 
# then pushing a new tag will not trigger a new evaluation.
```


You now should be able to see the details of your submission at : 
[gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/learning-to-smell-starter-kit/issues](gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/learning-to-smell-starter-kit/issues)


**Best of Luck**

# Author
[S.P. Mohanty](https://twitter.com/memohanty) <mohanty@aicrowd.com>

[Shivam Khandelwal](https://twitter.com/skbly7) <shivam@aicrowd.com>

