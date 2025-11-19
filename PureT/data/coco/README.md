# COCO Caption Annotations

The training and validation caption JSON files are not tracked to keep the
repository lightweight. Download the official COCO annotations and place them
here before training:

```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip annotations/captions_train2014.json \
    annotations/captions_val2014.json
cp annotations/captions_train2014.json PureT/data/coco/captions_train.json
cp annotations/captions_val2014.json PureT/data/coco/captions_validation.json
```

Adjust the paths if you already have the dataset elsewhere. Only the test JSON
included in this repo is kept for reference.
