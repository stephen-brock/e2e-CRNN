To run the model, execute train_model.sh

The Spectrogram uses torchaudio's MEL-Spectrogram, which may be broken on BlueCrystal.
If this is unable to be fixed, edit train_model.sh to remove --spectrogram:

$- python train_model.py --length 256 --stride 256 --epochs 40 --learning-rate 7e-4 --gamma 0.95 --dropout --norm

--length: Length of convolutional kernel or window length of spectrogram
--stride: Stride of convolutional kernel or hop length of spectrogram
--epochs: Number of training epochs
--learning-rate: Initial learning rate
--gamma: Gamma of exponential learning rate scheduler
--dropout: Adds dropout
--norm: Adds batch normalisation
--spectrogram: Uses spectrogram rather than raw audio

--dataset-root: Root of MagnaTagATune dataset
--log-dir: Root of log output directory
--batch-size: Size of batches
--val-frequency: How often does validation occur (epochs)
--log-frequency: How often is there an output to log (steps)
--print-frequency: How often is there an output to stdout (steps)
--worker-count: How many workers for loading the dataset