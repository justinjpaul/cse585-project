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
[Provided GH info](https://github.com/mosharaf/cse585/tree/f24/Resources/Starting%20with%20Cloudlab)

I found the remote ssh hostname by going to the manifest in an experiment and finding the following lines toward the top of the xml file. 
```bash
<login authentication="ssh-keys" hostname="hp167.utah.cloudlab.us" port="22" username="<username>"/>
```

So in this case ```ssh <username>@hp167.utah.cloudlab.us```