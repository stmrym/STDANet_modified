import glob
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator



# Specify the path for the new event file
new_event_file = "./exp_log/train/debug_2023-12-19T184932_STDAN_Stack_GOPRO/logs/train"

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
            
            if tag == 'Loss/EpochDeblurLoss_TRAIN':
                new_tag = 'Loss_TRAIN/DeblurLoss'
            elif tag == 'Loss/EpochMSELoss_TRAIN':
                new_tag = 'Loss_TRAIN/MSELoss'
            elif tag == 'Loss/EpochWarpMSELoss_TRAIN':
                new_tag = 'Loss_TRAIN/WarpMSELoss'
            # elif tag == 'PSNR/Epoch_PSNR_TRAIN':
                # new_tag = 'PSNR/TRAIN'

            if tag == 'Loss/EpochDeblurLoss_VAL':
                new_tag = 'Loss_VALID/DeblurLoss_BSD_3ms24ms'
            elif tag == 'Loss/EpochMSELoss_TRAIN':
                new_tag = 'Loss_VALID/MSELoss_BSD_3ms24ms'
            elif tag == 'Loss/EpochWarpMSELoss_TRAIN':
                new_tag = 'Loss_VALID/WarpMSELoss_BSD_3ms24ms'
            elif tag == 'PSNR/Epoch_PSNR_VAL':
                new_tag = 'PSNR/VAL_BSD_3ms24ms'

            for value in value_list:
                tf.summary.scalar(new_tag, value.value, step=value.step)