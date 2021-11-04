[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4291233.svg)](https://doi.org/10.5281/zenodo.4291233)

# Finicky Transfer Learning

<p align="center">
  <img src="https://i.ibb.co/82dqbV7/FTL-logo.png" alt="FTL project logo" width="375" height="361"/>
</p>

>**Finicky Transfer Leraning is an open source CNN pruning set of methods developed to enable widespread use of AI methods in areas with shortege of computing resources and edge-computing devices. It allows for transfer learning of CNN models trained on vast datasets to work with specific use cases while significantly reducing their inference time and increasing accuracy.**

## Table of contents
* [General info](#general-info)
* [Dependencies and setup](#dependencies)
* [Usage](#usage)
* [Use examples](#use-examples)
* [Acknowledgments](#acknowledgments)

## General info

Finicky Transfer Learning methods were made to enable the use of AI methods in underfunded governmental agencies with limited access to computing resources. Provided methods are meant to use for lowering inference time of image classification models used in infrastructure maintenance agencies for assessing defects in public infrastructure.

The main differences between regular Transfer Learning, CNN pruning and Finicky Transfer Learning are shown in the image below:

<p align="center">
  <img src="https://i.ibb.co/bFMFZ0Q/schemat-dla-mtxa.jpg" alt="FTL visual abstract" width="550" height="700"/>
</p>

## Dependencies and setup

With current release, Finicky Transfer Learning methods are provided for and tested on Linux based operating systems, however Windows OS is also supported if appropriate dependencies are fulfilled. 

All of the dependencies required to run repository methods are listed in `install_dependencies.sh` bash script. It can also be used to install required libraries directly on the machine. Note that for the package to run, at least `Python 3.6` is required.

## Usage

To use FTL methods, certain workflow should be maintained. 

First, segmented dataset should be provides, using labeler tool in `./Labeler/labeler.py`.

After obraining images with sought objects segmented out, they should be placed in `./Fussy TL/Segmented` and `Images` respectively.

Then, full CNN feature extractor backbone network should be placed in `./Fussy TL/backbones` for pruning.

In order to obtain pruned networks provided scripts should be run in order according to `./Fussy TL/workflow.sh`:
* make_fussy_models.py
* extract_features.py 
* train_models.py 
* test_models.py - for testing obtained models

Mentioned scripts apart from performing the default operations, can accept certain parameters and can be run directly from bash:
* make_fussy_models.py \<\<output models dir\>\> \<\<path to backbone CNN\>\>
* extract_features.py \<\<output models dir\>\> 
* train_models.py \<\<output models dir\>\> 
* test_models.py \<\<output models dir\>\> 
  
Pruned models will be saved in `./Fussy TL/ models` dir to be used in model deployment.

## Use examples

Ready to use pruning examples for crack recognition are available i n`./Fussy TL/models`. 

Specific models are derived from 100 models metrics made for both VGG16 vanilla model trained on ImageNet, and AlexNet trained from scratch on crack datset.

Models metrics:

<p align="center">
  <img src="https://i.ibb.co/zm0vHxL/metrics.png" alt="FTL metrics" width="937" height="495"/>
</p>

Decision function:

<p align="center">
  <img src="https://i.ibb.co/CnytBjP/dec-func.png" alt="FTL decision" width="530" height="350"/>
</p>

Models obtained using Finicky Transfer Learning Methods obtain higher accuracy and lower inference time as compared to base models. Comparison between base model and Finicky Transfer Learned one can be seen below (for AlexNet model, 5 consecutive runs have standard deviation in parentheses calculated covering all tested, initial model variants):

<p align="center">
  <img src="https://i.ibb.co/njm3YNp/Comparison.png" alt="FTL comparison" width="404" height="237"/>
</p>

## Acknowledgments

Finicky Transfer Learning is associated with research paper:

Mateusz Żarski<sup>[1](#footnote1)</sup>, Bartosz Wójcik<sup>[1](#footnote1)</sup>, Jarosław Adam Miszczak<sup>[2](#footnote2)</sup>, Kamil Książek<sup>[2](#footnote2)</sup>, *Finicky Transfer Learning - a method of pruning convolutional neural networks for cracks classification*, Computer-Aided Civil And Infrastructure Engineering, 2021, http://doi.org/10.1111/mice.12755, arXiv:[placeholder](placeholder) 

<a name="footnote1"><sup>1</sup></a>Department of Civil Engineering, Silesian University of Technology,
<a name="footnote2"><sup>3</sup></a>Institute of Theoretical and Applied Informatics, Polish Academy of Sciences

Finicky Transfer Learning is free to use under [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)

*This project was supported by Polish National Center for Research and Developement
under grant number POWR.03.05.00-00.z098/17-00.*
