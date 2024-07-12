put dataset to ./road_segmentation

1.run gen_csv.py to generate split csv file
2.run main.py to get test groundtruth
3.run submission_to_mask.py to get submission csv file


## Setup
Follow `https://auto.gluon.ai/stable/install.html` to install `autogluon`


## Run

```
python main.py

python submission_to_mask.py
```
