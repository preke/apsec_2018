# APSEC 2018

Codes and data for the paper **Detecting Duplicate Bug Reports with Convolutional Neural Network** in APSEC 2018

### Requirement:

- Python3.6
- Anaconda(numpy, pandas, sklearn)
- PyTorch 0.4.0
- torchtext
- gensim
- cuda 8.0



### Basic usage:

Before run codes, set parameters(paths) in each .py file in `codes/`
#### Traditional CNN

- Train model: `python main.py`
- Evaluate existed model:`python main.py -snapshot *.pt` (*.pt is the existed model)

#### DBR-CNN
- Generate DBR-CNN result: get into `codes/[data_set]/`[data_set] means the specific dataset you are using, like (spark, hadoop, hdfs, mapreduce)
- `python cb.py`

### Use specific word vectors:

In `main.py`:
1. set `use_global_w2v = False`
2. set `wordvec_save` to specific .save file

### Change CNN parameters:
In `main.py`:
change variables straightly in `parser`

---

Pretrained word vectors could be download from https://pan.baidu.com/s/18R_lZhlOdp-kgDlbrBq7iA
and unzip it in `codes/`
