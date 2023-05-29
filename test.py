
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.logger import Logger
import time

if __name__ == '__main__':
    opt, model_config = TestOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of training samples = %d' % dataset_size)

    model = create_model(opt, model_config)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    total_iters = 0                # the total number of training iterations

    model.eval()

    val_start_time = time.time()
    for i, data in enumerate(dataset):  # inner loop within the test dataset
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        model.cache_results()  # store current batch results

    model.compute_visuals()  # visualization
    t_val = time.time() - val_start_time

    model.compute_metrics()
    metrics = model.get_current_metrics()
    visualizer.print_current_metrics(-1, total_iters, metrics, t_val)

    model.save_data()