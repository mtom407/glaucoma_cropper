# glaucoma_cropper
Extract region of interest from funduscamera images

Part of the final engingeering project from Warsaw's University of Technology, Faculty of Electronics and Information Technology

This repository holds Python files implemented for preprocessing of funduscamera images in order to build a galucoma classifier. The algorithm itself finds swarms of max intensity pixels on a few different colorspaces and then uses these points to define the optic disc center with high accuracy. 

On the data used for this project presented algorithm achieved 96,69% accuracy. The cropping result was deemed accurate when the optic disc was present in the result image in its enitrety or at least the large portion of the disc was present.

Publication link: [to be added]

Example of the results gotten with the algorithm:

![Healthy](https://github.com/mtom407/glaucoma_cropper/blob/main/docs/images/example_1.png) ![Unhealthy](https://github.com/mtom407/glaucoma_cropper/blob/main/docs/images/example_2.png)


