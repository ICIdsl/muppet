**Multi-Precision Policy Enforced Training (MuPPET)** 
==============================
This is the open-sourced implementation of multi-precision training connected to the paper "Multi-Precision Policy Enforced Training (MuPPET): A precision-switching strategy for quantised fixed-point training of CNNs" that was published in ICML 2020. If you reference this work in a publication, we would appreciate you using the following citation:  
```
@misc{rajagopal2020multiprecision,
      title={Multi-Precision Policy Enforced Training (MuPPET): A precision-switching strategy for quantised fixed-point training of CNNs}, 
      author={Aditya Rajagopal and Diederik Adriaan Vink and Stylianos I. Venieris and Christos-Savvas Bouganis},
      year={2020},
      eprint={2006.09049},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Checkout https://www.imperial.ac.uk/intelligent-digital-systems to see other publications by the Intelligent Digital Systems Lab at Imperial College London.  

Installation
------------
```
git clone https://github.com/ICIdsl/pytorch_training.git
cd pytorch_training
git submodule update --init src/muppet 
```

The following section defines MuPPET specific config file parameters. 
config.ini
----------
> MuPPET\_Hyperparameters
- **Run\_Muppet** : If False, all following parameters are ignored and regular training is performed 
- **Bit\_Width** : Bitwidth at which MuPPET training begins. If FP32, set to -1 
- **Data\_Type** : One of "DFixed" or "Float". Has to match **Bit\_Width** specified
- **Round\_Meth** : One of "Simple" or "Stochastic"
- **Policy\_Resolution** : Refer to resolution hyperparameter in associated paper  
- **Policy\_Patience** : Refer to patience hyperparameter in associated paper
- **Fp32\_Epochs\_Per\_Lr** : Number of epochs run at each learning rate once in FP32 training  
- **Prec\_Schedule** : Precisions to change into at each switch. First precision must match with the value for **Bit\_Width**
