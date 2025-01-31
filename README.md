# NeuralOperatorDeepforReactionDiffusionPDEswithDelay
The source code for the paper is named Deep Learning of Delay-Compensated Backstepping for   Reaction-Diffusion PDEs. All the code is written in Python and relies on standard packages such as numpy, Pytorch, and the deep learning package DeepXDE.

# Step 1： Generation Data
Please see the file in the folder named data_genaration_k_Gamma_Gammay.py. This model can used to genarate data for trainning all kernels. 

# Step 2： Learning kernels
Please see the file in the folder named parablic_PDE_trainning_with_delay.py. This model can used to train all kernels.  

# Step 3： Simulation examples
Please see the file in the folder named parabolic_PDE_with_delay_DeepONet.py. This file can generate the dataset for Steps 1 and 2, and also can use train models for two simulation examples of Reaction-Diffusion PDEs with input delay. 

# Questions
Feel free to leave any questions in the issues of Github or email at wss_dhu@126.com.
