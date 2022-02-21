# Circuit-specific flickering of sleep and wake predicts natural behaviors

This git repo contains the code base developed to run the experiments for the paper listed in the title. As of February 2022, the paper has been submitted and is undergoing review. When the paper is available this repo will be updated with details about the paper. Until this this repo will remain available but without reference to the original paper.

## Directory Structure

```
src/                      # source code, includes model train, evaluate, and test code
 ├» scripts/              # utilities, model labeling, optical flow, visualization tools
 ├» parameter_files/      # model, experiment, and dataset parameters
 ├» models/               # neural network model definition
 └» common_py_utils/      # common python utilities, yaml parser, etc
docker/                   # Dockerfile defines the docker container environment all code runs within
k8s/                      # Kubenetes yaml and job submission scripts used to deploy code on the PRP cluster
requirements.txt          # Python dependencies which are installed as part of the Dockerfile build process
README.md                 # This documentation, feel free to contact us via github issues
```

## Pacific Research Platform (PRP)

https://pacificresearchplatform.org/

The primary compute and storage was performed on the PRP, a shared academic compute cluster. A multitude of institutions are granted shared access to the cluster.

## Contact Us

Feel free to contact us via Github Issues on this page.