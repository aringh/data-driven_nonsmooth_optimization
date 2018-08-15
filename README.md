Data-drive nonsmooth optimization
=====================================

This repository contains the code for the article "Data-driven nonsmooth optimization" by S. Banert, A. Ringh, J. Adler, J. Karlsson, and O. Öktem. An arxiv version of the article can be found [here](https://arxiv.org/abs/1808.00946).

Contents
--------
The code contains the following

* Files used for training the algorithm.
* Files used for validation, both generate one slice/reconstruction and objective function values on a batch of data.
* Files for generalization to deconvolution.

Note that the Mayo Clinic data used in the training do not belong to the authors and must therefore be obtained separately, see [here](https://www.aapm.org/GrandChallenge/LowDoseCT/).


Installing and running the code
-------------------------------
Clone this repository, and the [ODL repository](https://github.com/odlgroup/odl).
Install ODL from source, e.g., by following the [ODL installation instructions](https://odlgroup.github.io/odl/getting_started/installing.html).
Also install numpy, scipy, and ASTRA (version 1.8.3).
If you are using conda, the latter can be installed with the following command
* $ conda install -c astra-toolbox astra-toolbox=1.8.3

After this, the scripts can be run using, e.g., spyder.
Training has been done using ODL commit 0ab389f, and validation one done using ODL commit eff7129.


Contact
-------
[Sebastian Banert](https://www.kth.se/profile/banert), Postdoc  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
banert@kth.se

[Axel Ringh](https://www.kth.se/profile/aringh), PhD student  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
aringh@kth.se

[Jonas Adler](https://www.kth.se/profile/jonasadl), PhD student  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
Elekta Instrument AB, Stockholm, Sweden  
jonasadl@kth.se

[Johan Karlsson](http://math.kth.se/~johan79), Associate Professor  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
johan.karlsson@math.kth.se

[Ozan Öktem](https://www.kth.se/profile/ozan), Associate Professor  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
ozan@kth.se


Funding
-------
We acknowledge Swedish Foundation of Strategic Research grants AM13-0049 and ID14-0055, Swedish Research Council grant 2014-5870 and support from [Elekta](https://www.elekta.com/).

The authors thank Dr. Cynthia McCollough, the Mayo Clinic, and the American Association of Physicists in Medicine for providing the data necessary for performing comparison using a human phantom.
