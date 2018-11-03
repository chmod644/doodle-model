# Model and Tools for Quick, Draw! Doodle Recognition Challenge 

## Usage

### Preprocess

```sh
# Full dataset
./split_chunk.py --input ../input/train_simplefield/ --output ../input/train_simplefied_chunk

# Limit number of elements in each category
./split_chunk.py --input ../input/train_simplefield/ --output ../input/train_simplefied_chunk120000 --max_element 120000

# Create debug dataset (100 elements in each category)
./split_chunk.py --input ../input/train_simplefield/ --output ../input/train_simplefied_debug --debug
```

output the files below

```
../input/train_simplefied_chunk
|-- train_k0.csv
|-- train_k1.csv
...
|-- train_k98.csv
`-- train_k99.csv
```

### Training

```sh
# NOTE: Specify save interval as step
./train.py --input ../input/train_simplefied_chunk --archi <original|resnet34|resnet50> --epochs 20 --save_interval 100000

# Show training image, not train
./train.py --input ../input/train_simplefied_chunk --debug

# Show help message
./train.py --helpfull
```


output the files below

```
../output/
`-- model
    |-- 10000.pth
    |-- 20000.pth
    ...
    |-- 200000.pth
    `-- model.txt
```


### Validation
```sh
# Load latest model
./validation.py --input ../input/train_simplefied_chunk --archi <original|resnet34|resnet50> --model ../output/model

# Specify the model
./validation.py --input ../input/train_simplefied_chunk --archi <original|resnet34|resnet50> --model ../output/model/10000.pth
```

### Create submittion file

```sh
# Load latest model
./submit.py --input ../input/test_simplefied.csv --archi <original|resnet34|resnet50> --model ../output/model

# Specify the model
./submit.py --input ../input/test_simplefied.csv --archi <original|resnet34|resnet50> --model ../output/model/10000.pth
```

output the files below

```
../output/
|-- inference.pickle
`-- submission.csv
```
