# SRToANet


## Data Preperation

All the information of channels (time delays, complex attenuation) are stored in `data/Pathset_train.mat`, `data/Pathset_test.mat`, and `data/Pathset_test_802.mat`

Simply run `data/CIR_Generation.m` will generate the dataset for training and testing, stored in `data/traindata` and `data/testdata` respectively

## Train the model

Run `train.py` to train the model. It will first train the super-resolution network and then the two regressors

    usage: train.py [-h] [--name NAME] [--snr {high,low}] [--bandwidth BANDWIDTH]
                [--e_sr E_SR] [--lr_sr LR_SR] [--batch_sr BATCH_SR]
                [--lam LAM] [--up UP] [--e_reg E_REG] [--lr_reg LR_REG]
                [--batch_reg BATCH_REG] [--use_ori [USE_ORI]]
                [--window_len WINDOW_LEN] [--window_index WINDOW_INDEX]
                [--device {cpu,gpu}] [--print_interval PRINT_INTERVAL]

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           name for the experiment folder
      --snr {high,low}      select high SNR scenario or low SNR scenario
      --bandwidth BANDWIDTH
                        define the system bandwidth
      --e_sr E_SR           number of epochs to train the super resolution network
      --lr_sr LR_SR         learning rate for the super resolution network
      --batch_sr BATCH_SR   batchsize for the super resolution network
      --lam LAM             weight for the time domain loss
      --up UP               upsamping rate for the super resolution net
      --e_reg E_REG         number of epochs to train the super resolution network
      --lr_reg LR_REG       learning rate for the super resolution network
      --batch_reg BATCH_REG
                        batchsize for the super resolution network
      --use_ori [USE_ORI]   whether to include the original observation for the
                        regressors
      --window_len WINDOW_LEN
                        window size for the second regressor
      --window_index WINDOW_INDEX
                        number of samples for the window for the second
                        regressor
      --device {cpu,gpu}    choose the device to run the code
      --print_interval PRINT_INTERVAL
                        number of iterations between each loss print
## Test 

Run `test.py` to test the model. You can test the customized channel model and the 802.15.4a channel model.

    usage: test.py [-h] [--name NAME] [--snr SNR] [--bandwidth BANDWIDTH]
               [--up UP] [--use_ori [USE_ORI]] [--use_802 [USE_802]]
               [--window_len WINDOW_LEN] [--window_index WINDOW_INDEX]
               [--device {cpu,gpu}] [--num_test NUM_TEST]

    optional arguments:
      -h, --help            show this help message and exit
      --name NAME           name for the experiment folder
      --snr SNR             select high SNR scenario or low SNR scenario
      --bandwidth BANDWIDTH
                        define the system bandwidth
      --up UP               upsamping rate for the super resolution net
      --use_ori [USE_ORI]   whether to include the original observation for the
                        regressors
      --use_802 [USE_802]   whether to test the 802.15.4a channel
      --window_len WINDOW_LEN
                        window size for the second regressor
      --window_index WINDOW_INDEX
                        number of samples for the window for the second
                        regressor
      --device {cpu,gpu}    choose the device to run the code
      --num_test NUM_TEST   number of test cirs
