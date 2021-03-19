# decagon_internship
Temporary repository of Decagon for summer internship applications

Dowload data folder from [google drive](https://drive.google.com/drive/u/0/folders/1wspQAqZ8Ulbry1xZcUP9s8C8ILiO7Xrl)

To run trained model on test data:
```
$ python inference.py --num_workers 0 --gpu --real --drug_embed_mode one-hot --checkpoint_path 'data/model_weights/weights_from_onehot.pt'
```

To train new model
```
$ python decagon_run.py --num_workers 0 --gpu --drug_embed_mode 'one-hot' --real --print_progress_every 10
```

Also you can train model on synthetic data
```
$ python decagon_run.py --num_workers 0 --gpu --drug_embed_mode 'one-hot' --print_progress_every 10
```
