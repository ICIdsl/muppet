import sys
import csv

import configparser as cp
import numpy as np

def main(fileName, epochList, bitWidthList, dataTypeList):
    #{{{
    for count, bitWidth in enumerate(bitWidthList):
        dataType = dataTypeList[count]
        for count, epoch in enumerate(epochList):
            net = "resent"
            dataset = "cifar100"
            model = "{}_{}".format(net, dataset)
            targetEpoch = int(epoch) + 9
            target = {
                'dataset': "cifar100",
                'dataset_location': "/data",

                'architecture': "resnet",

                'print_only': 'False',
                'total_epochs': '{}'.format(targetEpoch),
                'train_batch': '128',
                'test_batch': '128',
                'learning_rate': '0.1',
                'min_lr': '0.001',
                'gamma': '0.1',
                'momentum': '0.9',
                'weight_decay': '1e-4',
                'lr_schedule': '50 -1 100 -1',
                'train_val_split': '0.8',

                'run_muppet': 'True',
                'bit_width': "{}".format(bitWidth),
                'data_type': "{}".format(dataType),
                'round_meth': 'Stochastic',
                'prec_schedule': "{}".format(bitWidth),

                'manual_seed': '{}'.format(seed[count]),
                'gpu_id': '3',
                'checkpoint_path': "/home/dav114/pytorch_training/src/dav114/hamm_train/logs",
                'test_name': "{}_test".format(model),
                'pretrained': "/home/dav114/pytorch_training/src/muppet/logs/resnet_fp32_cifar100/2019-11-12-17-07-27/orig/{}-model.pth.tar".format(epoch),
                'resume': 'False',
                'branch': 'True',
                'evaluate': 'False',
            }

            config = cp.ConfigParser()

            config.read(fileName)
            for configKey, configData in config.items():
                for dataKey, data in configData.items():
                    if dataKey in target.keys():
                        config[configKey][dataKey] = target[dataKey]

            if bitWidth == '-1':
                bitWidth = '32'
            outFileName = "{}/pvalue/{}-{}bit-{}seed.ini".format(dataset, epoch, bitWidth, seed[count])
            with open(outFileName, 'w') as outFile:
                config.write(outFile)
            if bitWidth == '32':
                bitWidth = '-1'
    #}}}

if __name__ == '__main__':
#{{{
    fileName = sys.argv[1]
    bitWidths = ['-1', '8', '12', '14', '16']
    dataTypes = ['Float', 'DFixed', 'DFixed', 'DFixed', 'DFixed']
    epochs = ['9', '40', '30', '30', '30', '30', '30']
    seed = ['4', '10', '13', '12', '8', '34', '22']
    main(fileName, epochs, bitWidths, dataTypes)
#}}}
