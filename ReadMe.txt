This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] Linglong Dai, Xiuhong Wei, "Distributed machine learning based downlink channel estimation for RIS assisted wireless communications",  IEEE Trans. Commun., 2022.

*********************************************************************************************************************************
If you use this simulation code package in any way, please cite the original paper [1] above. 
 
The author in charge of this simulation code pacakge is: Xiuhong Wei (email: weixh19@mails.tsinghua.edu.cn).


Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers (more information can be found at: 
http://oa.ee.tsinghua.edu.cn/dailinglong/publications/publications.html). 

Please note that the Python 3.6 and Pytorch 1.6.0 are used for this simulation code package,  and there may be some imcompatibility problems among different versions. 

Copyright reserved by the Broadband Communications and Signal Processing Laboratory (led by Dr. Linglong Dai), Beijing National Research Center for Information Science and Technology (BNRist), Department of Electronic Engineering, Tsinghua University, Beijing 100084, China. 

*********************************************************************************************************************************
Abstract of the paper: 

The downlink channel estimation requires a huge pilot overhead in the reconfigurable intelligent surface (RIS) assisted communication system. By exploiting the powerful learning ability of the neural network, the machine learning (ML) technique can be used to estimate the high-dimensional channel from a few received pilot signals at the user. However, since the training dataset collected by the single user only contains the information of part of the channel scenarios of a cell, the neural network trained by the single user is not able to work when the user moves from one channel scenario to another. To solve this challenge, we propose to leverage the distributed machine learning (DML) technique to enable the reliable downlink channel estimation. Specifically, we firstly build a downlink channel estimation neural network shared by all users, which can be collaboratively trained by the BS and the users with the help of the DML technique. Then, we further propose a hierarchical neural network architecture to improve the channel estimation accuracy, which can extract different channel features for different channel scenarios. Simulation results show that compared with the neural network trained by the single user, the proposed DML based neural networks can achieve better estimation performance with the reduced pilot overhead for all users from different scenarios.

*********************************************************************************************************************************
How to use this simulation code package?

We have considered two types of channel samples, i.e., Saleh-Valenzuela channel model and the publicly-available DeepMIMO dataset based on ray-tracing.

(1) Open folder "CE_SV" and run "Test.py",  you will obtain the Fig. 4 and Fig. 5 in turn by using our trained models. The generated simultion results will be saved the folder named 'results'.

(2) Open folder "CE_DeepMIMO" and run "Test.py",  you will obtain the Fig. 6 and Fig. 7 in turn by using our trained models. The generated simultion results will be saved the folder named 'results'.

(3) If you want to retrain the models, please run the following steps in the folder "CE_SV" and folder "CE_DeepMIMO":

a) Run generate_data.py, and you will generate the training data, which will be saved in the folder named 'available_data';

b) Run Runner_P128.py, and you will train the DCE network and HDCE network for the pilot overhead Q=128.  Change the value  of the "order_number", you will train the network according to different training methods. The trained network will be saved in the folder named 'workspace/Pn_128'.


It is noted that there may be some differences in the results of different training processes. 

*********************************************************************************************************************************
Enjoy the reproducible research!