# KDD'23 --- Graph Neural Processes for Spatio-Temporal Extrapolation
Official code of the paper 'Graph Neural Processes for Spatio-Temporal Extrapolation'.

**Abstract:** 
We study the task of spatio-temporal extrapolation that generates data at a target location from surrounding contexts structured in a graph. 
This task is crucial as sensors that collect data are sparsely deployed, resulting in a lack of fine-grained information due to high deployment and maintenance costs. 
Existing methods either use learning-based models like Neural Networks or statistical approaches like Gaussian Processes for this task. However, the former falls short in providing uncertainty estimates while the latter fails to capture complex spatial and temporal correlations in data effectively.
To address these issues, we propose Spatio-Temporal Graph Neural Processes (STGNP), a neural latent variable model which commands these capabilities simultaneously. Specifically, we first learn deterministic spatio-temporal representations by stacking layers of causal convolutions and cross-set graph neural networks.
Then, we learn latent variables for target locations through vertical latent state transitions along layers and obtain extrapolations. 
Importantly during the transitions, we propose Graph Bayesian Aggregation (GBA), a Bayesian graph aggregator that aggregates contexts considering uncertainties in the context data and the graph structure.
Extensive experiments demonstrate that STGNP has desirable properties such as uncertainty estimates and high learning capabilities, and achieves state-of-the-art results by a clear margin.
## Requirements
- [torch](https://pytorch.org/)
- numpy
- pandas
- [neptune](https://neptune.ai/) (optional)

To install requirements (with out neptune):
```bash
pip install -r requirements.txt
```

## Datasets
* Beijing: The original dataset is accessible at [this link](https://www.microsoft.com/en-us/research/publication/forecasting-fine-grained-air-quality-based-on-big-data/?from=http%3A%2F%2Fresearch.microsoft.com%2Fapps%2Fpubs%2F%3Fid%3D246398). 
The processed data for our project is available at [Google Drive](https://drive.google.com/drive/folders/13lpPTn0XYxETPGVEi9gUX09Tr3vWwHVK?usp=sharing). 
* London: The dataset can be downloaded at [KDD 18 CUP](https://www.biendata.xyz/competition/kdd_2018/). Note that registration is required.
* Water: The dataset can be accessed by contacting the authors of the [TDB paper](http://urban-computing.com/pdf/ieeetbd2020_UrbanWater.pdf).

Please put the processed data under [dataset](/data) folder.

## Train / Evaluate STGNP
To train and evaluate models, please run the following command:
```bash
./train.sh [model] [dataset] [attribute] [config] [gpu_ids] [seed]
```

| setting   | values                                                        | help                                                                                            |
|-----------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| model     | hierarchical                                                  | model name, 'hierarchical' means our STGNP.                                                     |
| dataset   | BJAir, Water, LDAir, BJAirDEP                                 | Beijing air quality dataset, DEP is used for dense extrapolation for viauslization in Figure 1. |
| attribute | PM25_Concentration, PM10_Concentration, NO2_Concentration, RC | RC is for the Water dataset                                                                     |
| config    | config1 (setting for the default results), config2, ...       | config files for models, see configurations in [model_configurations](model_configurations)     |
| gpu_ids   | 0, 1, 2, 3                                                    | gpu ids to use, as we use torch.distributions, multi-GPUs are not supported currently           |
| seed      | 0, 1, 2, 3, ...                                               | random seed for reproducibility                                                                 |

Each running will train the model 5 times independently with the random seed increasing by 1 each time. 
The framework will save the best model with the highest validation accuracy and evaluate it on the test set automatically after training. 
All the checkpoints and results will be saved at [checkpoints](checkpoints) folder.
For more training, testing, dataset configurations, please refer to [base_options](options/base_options.py), [train_options](options/train_options.py), [test_options](options/test_options.py), and [dataset_options](options/dataset_options.py).

## Reproduce Our Results
We saved the pretrained checkpoints for our STGNP at [Google Drive](https://drive.google.com/drive/folders/13lpPTn0XYxETPGVEi9gUX09Tr3vWwHVK?usp=sharing).
Download the checkpoints files and put them under [checkpoints](checkpoints) folder.
Each checkpoint file contains *run_test.sh* script.
Please run the script **in the project root folder** to reproduce our results:
```bash[README.md](..%2F..%2F..%2F..%2FDesktop%2FREADME.md)
chmod u+x run_test.sh
./run_test.sh
```
The numerical results will be saved at *metrics.txt* and printed out.
The extrapolation results, ground truth, and the uncertainty estimates (if applicable) will be saved at *results.pkl*.

## Acknowledgment
The code framework is partly adopted from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) as they provide functions to load, save, and optimize models. Thanks for their great work!!

## Citation
If you find our model useful, please cite our paper:
```
@inproceedings{hu2023graph,
  title = {Graph Neural Processes for Spatio-Temporal Extrapolation},
  author = {Hu, Junfeng and Liang, Yuxuan and Fan, Zhencheng and Chen, Hongyang and Zheng, Yu and Zimmermann, Roger},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year = {2023}
}
```
## Questions
If you have any questions, please contact me at junfengh@u.nus.edu.
