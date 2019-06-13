# Dangerous Driving Classification

## Setup
****
This environment has only been tested in Python 3, so I highly recommend you to use Python 3 to run these scripts. I also highly recommend you to use virtual environment to run these scripts. You can create a virtual environment with this command:

```
python3 -m venv env
```
or 
```
python -m venv env
```

Then you can use that virtual environment with this command:

for OSX or Linux:
```
$ source env/bin/activate
```

for Windows:
```
env\Scripts\activate
```

This project has several dependencies. You need to install these dependencies in order to run the scripts successfully. Use this command to install dependencies:
```
python -m pip install -r requirements.txt
```

****
## Running Scripts
***

There are 3 main scripts in this project.

- [predict](predict.py) script

    In order to run the predict script, **make sure that the models are available in models folder**. 
    
    If the models are already available, you can the run the script by:
    ```
    python predict.py TRIP_FILE_1 TRIP_FILE_2 TRIP_FILE_3 ... TRIP_FILE_N
    ```
    - TRIP_FILE_N = Path of dataset. The dataset can be separated into multiple parts.

    Example:
    ```
    python predict.py test_part_1.csv test_part_2.csv test_part_3.csv
    ```

    **However, if there are no models available in models folder, you need to run the [train script](train.py) first.**

- [train](train.py) script

    You can use this command to train and save the machine learning models. Use this command to run the script:

    ```
    python train.py
    ```
    **Please make sure that the [safety dataset](https://www.aiforsea.com/safety) directory is in the same directory as this script. This script is only designed to train from the safety dataset that Grab provided**

- [score](score.py) script

    In order to run the score script, **make sure that the models are available in models folder**. 
    
    If the models are already available, you can the run the script by:
    ```
    python predict.py TRIP_FILE_1 TRIP_FILE_2 TRIP_FILE_3 ... TRIP_FILE_N LABEL_FILE
    ```
    - TRIP_FILE_N = Path of dataset. The dataset can be separated into multiple parts.
    - LABEL_FILE = Path of label. There's only one label file that can be loaded into this script.

    Example:
    ```
    python predict.py test_part_1.csv test_part_2.csv test_part_3.csv label.csv
    ```

    **However, if there are no models available in models folder, you need to run the [train script](train.py) first.**