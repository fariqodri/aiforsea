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

    In order to run the predict script, **make sure that the models are available in models folder. If there are no models available in models folder, you need to run the [train script](train.py) first.**. 
    
    If the models are already available, you can the run the script by:
    ```
    python predict.py TRIP_FILE_1 TRIP_FILE_2 TRIP_FILE_3 ... TRIP_FILE_N
    ```
    - TRIP_FILE_N = Path of dataset. The dataset can be separated into multiple parts.

    Example:
    ```
    python predict.py test_part_1.csv test_part_2.csv test_part_3.csv
    ```

    <small>I already provided [test.csv](test.csv) if you want to try this script immediately.</small>

- [train](train.py) script

    You can use this command to train and save the machine learning models. Use this command to run the script:

    ```
    python train.py
    ```
    **Please make sure that the [safety dataset](https://www.aiforsea.com/safety) directory is in the same directory as this script. This script is only designed to train from the safety dataset that Grab provided**

- [score](score.py) script

    In order to run the score script, **make sure that the models are available in models folder. If there are no models available in models folder, you need to run the [train script](train.py) first.**. 
    
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

***
## How It Works
***
    
The first feature extraction is how to change the sensor reading over time into meaningful features. These sensor readings can be extracted into driving event (steady, hard turn, hard acceleration/deceleration). However, the biggest challenge of this task is there's no ground truth to compare to. So I came up with an idea to use K-Means clustering and sliding window algorithm.

#### Clustering

For this task, I fetch the trip with the longest time. Then the **variance** of every sliding window in acceleration_y and gyro_z features is extracted. The size of the sliding window is 3 for these features. Then these variance will be fed into the K-Means model. After some comparison by [silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)), it was known that the best number of clusters for this data is 3. Then the K-Means model will be used later in counting how many events occurred for each event in every trip.

#### Classification

For every trip, the **variance** of every sliding window in acceleration_y and gyro_z features is extracted. Then I use the K-Means model to count how many events occurred during the trip using the variance of every window. These number of events for each event will be fed into the Random Forest Classifier.

The reason why variance is chosen is because it tells about how spread the data is. If the spread within that window is big, it means that the driver changed the acceleration rapidly or made hard turn during that time frame because the linear acceleration or the angular acceleration will change quite significantly.

For more information about the step-by-step process, please run the [Jupyter Notebook file](aiforsea.ipynb).