

### Datasets
3 Dataset just for Demo
`fwt_sub_test_wiki`, `fwt_sub_val_nyt`, `fwt_sub_val_wiki`.


### Feature encoder pre-training
We adopt `baseline++` for MatchingNet, and `baseline` from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) for other metric-based frameworks.

-Train pre-trained feature encoder (specify `PRETRAIN` to `baseline++` or `baseline`).
```
python3 train_baseline.py --method baseline --dataset fwt_sub_test_wiki --name PRETRAIN 
```

### Training with multiple seen domains
Baseline training w/o feature-wise transformations.
- `METHOD` : `matchingnet`, `relationnet_softmax`, or `gnnnet`, or `protonet`.
- `TESTSET`: unseen domain `fwt_sub_val_nyt`, `fwt_sub_val_wiki`.
```
python3 train_baseline.py --method protonet --dataset multi --testset TESTSET --name multi_TESTSET_ori_METHOD --warmup PRETRAIN
```
Training w/ learning-to-learned feature-wise transformations.
```
python3 train.py --method METHOD --dataset multi --testset TESTSET --name multi_TESTSET_lft_METHOD --warmup PRETRAIN
```

### Evaluation
Test the metric-based framework `METHOD` on the unseen domain `TESTSET`.
- Specify the saved model you want to evaluate with `--name` (e.g., `--name multi_TESTSET_lft_METHOD` from the above example).
```
python3 test.py --method METHOD --name NAME --dataset TESTSET
```

