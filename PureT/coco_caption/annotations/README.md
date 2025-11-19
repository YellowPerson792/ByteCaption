# COCO 2014 Validation Annotations

`captions_val2014.json` is required by the official COCO evaluation scripts but
is not tracked because it is ~30 MB. Download it from the official COCO
annotations release and copy it to this directory when running evaluation:

```
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip annotations/captions_val2014.json
cp annotations/captions_val2014.json PureT/coco_caption/annotations/
```

If you have the annotations elsewhere, simply symlink or copy the file here.
