# afqstensorflowutils

A set of unrelated helper functions for my TensorFlow codes. I'm interested in physics applications with continuous field inputs and outputs, so the networks I make look a little different from the more common image recognition problems.

rescale.py is a little script for chopping up and shuffling csv datasets:
```bash
python -m afqstensorutils.rescale  data_files/water_lg.csv 0 true
```