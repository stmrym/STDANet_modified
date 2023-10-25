import glob
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator



# Specify the path for the new event file
new_event_file = "./exp_log/train/2023-09-27T162314_STDAN_Stack_DVD/logs/test"

# Specify the path to the existing event file
original_event_files = sorted(glob.glob(new_event_file + '/*tfevents.*'))


for original_event_file in original_event_files:
    print(original_event_file)

    # Load the existing event file using EventAccumulator
    event_acc = event_accumulator.EventAccumulator(original_event_file)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]

    # Create a new event file and write the logged values into it
    with tf.summary.create_file_writer(new_event_file).as_default():
        for tag in tags:
            value_list = event_acc.Scalars(tag)
            
            if tag == 'EpochWarpMSELoss_TRAIN':
                new_tag = 'Loss/EpochMSELoss_TRAIN'
            elif tag == 'EpochMSELoss_TRAIN':
                new_tag = 'Loss/EpochWarpMSELoss_TRAIN'
            elif tag == 'EpochDeblurLoss_TRAIN':
                new_tag = 'Loss/EpochDeblurLoss_TRAIN'
            elif tag == 'EpochPSNR_TRAIN':
                new_tag = 'PSNR/Epoch_PSNR_TRAIN'

            elif tag == 'EpochWarpMSELoss_TEST':
                new_tag = 'Loss/EpochMSELoss_TEST'
            elif tag == 'EpochMSELoss_TEST':
                new_tag = 'Loss/EpochWarpMSELoss_TEST'
            elif tag == 'EpochDeblurLoss_TEST':
                new_tag = 'Loss/EpochDeblurLoss_TEST'
            elif tag == 'EpochPSNR_TEST':
                new_tag = 'PSNR/Epoch_PSNR_TEST'

            for value in value_list:
                tf.summary.scalar(new_tag, value.value, step=value.step)