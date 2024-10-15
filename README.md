# cse585-project
Final Project for CSE 585

## .env 

## Data processing
#### bin/download_data.py
`bin/download_data.py` script pulls HuggingFace's [chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/viewer/default/train?sort[column]=judge&sort[direction]=asc&sort[transform]=length)

`bin/download_data.py` downloads, formats, and writes cleaned query and response data from the `conversation_a` column (selected arbitrarily)

To run the script: 
* Create a huggingface account and download huggingface-CLI locally
* Log in to huggingface-CLI on command line
* Be sure to ```pip install datasets``` (Will add to requirements.txt later)
* Run `python bin/download_data.py`


## Cloudlab info

#### Hardware
VLLM is run on CUDA so we need a node with a GPU. The different kinds are listed [here](https://docs.cloudlab.us/hardware.html). Probably the best one for us to use is the `c240g5` in Wisconsin. This can be specified in the `Parameterize` step for the experiment.

#### VLLM build
This section is to get our [forked VLLM build](https://github.com/sastec17/585_vllm) onto the image. It is public so get the [https url](https://github.com/sastec17/585_vllm.git) and then 
```bash 
git clone https://github.com/sastec17/585_vllm.git
```

#### SSH info

EDIT: it is much easier to find this by going into `List View` and looking at the SSH command.

[Provided GH info](https://github.com/mosharaf/cse585/tree/f24/Resources/Starting%20with%20Cloudlab)
I found the remote ssh hostname by going to the manifest in an experiment and finding the following lines toward the top of the xml file. 
```bash
<login authentication="ssh-keys" hostname="hp167.utah.cloudlab.us" port="22" username="<username>"/>
```

So in this case ```ssh <username>@hp167.utah.cloudlab.us```

#### Setup image with Python
The images e.g. `Ubuntu 22.04` don't come with python or pip installed. You need to run some of this stuff for python/pip access.

```bash
sudo apt update
sudo apt install python3
sudo add-apt-repository universe
sudo apt install python3-pip
```

#### Installing conda and vLLM - SHOULD BE ABLE TO START HERE

IMPORTANT - Run on a GPU with compute capability of 7.0 or higher (see vLLM docs). The following are profiles I've cross-referenced w/Nvidia docs that should work:
* d8545
* nvidiagh
* c4130

See node availability [here](https://www.cloudlab.us/resinfo.php#)

I've been testing on c4130 on the Clemson cluster, but often requires some reservation in advance.

Note - There may be more profiles/nodes that I missed. Check out [Nvidia list](https://developer.nvidia.com/cuda-gpus) and cross reference with [CloudLab hardware](http://emulab.pages.flux.utah.edu/testbed-manual/cloudlab-manual/hardware.html). Not sure if that's a comprehensive Nvidia list atm

Download cuda version 12: 
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```
Note that the above steps are for Linux Ubuntu version 18.04. If running on a different configuration, check [Nvidia](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_network) 

Reboot the system with `sudo reboot` to gain access to cuda. This takes an annoying few minutes

Download miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Install miniconda:
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Activate miniconda:
```bash
source ~/miniconda3/bin/activate
```

Download vLLM (will take a few minutes):
```bash
pip install vllm
```

Download wheel for vLLM 
```bash
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

Download code from forked git repo:
```bash
git clone https://github.com/sastec17/585_vllm.git
cd 585_vllm
```

Run python build -> This makes it so anytime we run code that imports `vllm`, it will use our changes, rather than what was released by the devs - Pretty epic ngl :p
```bash
 python python_only_dev.py
```

Ideally we get this to run:
```bash
python examples/offline_inference.py 
```

Or we get something like this to run, which will serve vllm so we can access it via api call:
```bash
python -m vllm.entrypoints.api_server --model gpt2 --device cuda
```