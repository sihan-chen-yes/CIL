put original dataset to `./road_segmentation`


## Setup
Follow `https://auto.gluon.ai/stable/install.html` to install `autogluon`


## Run

### extra data collection

```python
python crawl_aerial_seg.py
```

extra data will be put to `./road_segmentation/collect`

### augmentation

```python
python data_aug.py
```

put the data which is intend to augment to `./road_segmentation/train_original`

augmentation data will be generate to `./road_segmentation/aug`

### MRF post processing

```python
python MRF_smooth.py
```

after generating images in `./road_segmentation/test/groundtruth`, run the script, smoothed groundtruth will be generated in  

`./road_segmentation/test/smooth_gt`

### train

```python
python gen_csv.py
python main.py
python submission_to_mask.py
```

