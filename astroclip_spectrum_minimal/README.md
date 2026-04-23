# AstroCLIP Spectrum Minimal Reproduction (Workstation Version)

This version is designed for workstation / HPC training with pure PyTorch, without Lightning or WandB.

## Features
- masked spectrum reconstruction training
- epoch-level train/validation loss curves
- automatic resume from `output_dir/last.pt`
- final test after training using the best checkpoint
- test reconstruction visualization
- export of intermediate embeddings for every transformer block
- export of all learnable parameters and parameter statistics

## Install
```bash
pip install torch torchvision pyyaml matplotlib datasets pyarrow numpy
```

## Run
```bash
python train.py --config config.yaml
```

To resume explicitly from a checkpoint:
```bash
python train.py --config config.yaml --resume outputs_specformer/last.pt
```

## Main outputs
Inside `output_dir` you will get:
- `last.pt`: latest training checkpoint
- `best.pt`: best validation checkpoint
- `epoch_XXX.pt`: epoch checkpoints
- `history.json`: train/val/test history
- `loss_curves.png`: train/val loss curve figure
- `val_reconstruction_epoch_XXX.png`: validation reconstruction plots
- `test_results/test_metrics.json`: final test metrics
- `test_results/test_reconstruction.png`: final test reconstruction plot
- `test_results/layer_embeddings.pt`: token embedding + every block output + final embedding
- `test_results/layer_embedding_summary.csv`: embedding shape and norm summary
- `test_results/model_state_dict.pt`: full model parameters
- `test_results/parameter_stats.csv`: parameter statistics for anomaly checking
- `test_results/parameters/*.pt`: one tensor file per parameter

## Notes on testing
This project is a self-supervised masked reconstruction model, so the final test is also evaluated with masked reconstruction loss on the test split. That is the correct task-aligned notion of "model effect" here.
