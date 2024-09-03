# CIL Road Segmentation Project



## Setup
- put original dataset to `./road_segmentation`
- Follow `https://auto.gluon.ai/stable/install.html` to install `autogluon`


## Run

### extra data collection

```python
python crawl_aerial_seg.py
```

extra data will be put to `./road_segmentation/collect`

then generate csv file for extra data
```python
python gen_csv_additional.py
```

<!-- ### augmentation

```python
python data_aug.py
```

put the data which is intend to augment to `./road_segmentation/train_original`

augmentation data will be generate to `./road_segmentation/aug` -->

### Train

```python
python gen_csv.py
python main_model_ensembles.py
python submission_to_mask.py
```


### Post processing
change the `post_process_method` to choose CRF/Graph-cut methods to do post-processing.
```python
python post_process.py
```

after generating images in `./road_segmentation/test/groundtruth`, run the script, smoothed groundtruth will be generated in  

`./road_segmentation/test/smooth_gt`


