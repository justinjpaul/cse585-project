# cse585-project
Final Project for CSE 585

## .env 

## Conda virtual environment guide (optional)
1. Install Anaconda or [Miniconda](https://docs.anaconda.com/miniconda/)
2. Run `conda -V` to verify installation worked
3. Run `conda create -n <yourenvname>` to create virtual environment 
4. Activate environment via `source activate <yourenvname>`
5. Install dependencies via `pip install -r requirements.txt` - Note, you may need to install `pip` or `pip3`

Develop / run within this virtual environment
Run `conda deactivate` to deactivate the environment

## Data processing
#### bin/download_data.py
`bin/download_data.py` script pulls HuggingFace's [chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations/viewer/default/train?sort[column]=judge&sort[direction]=asc&sort[transform]=length)

`bin/download_data.py` downloads, formats, and writes cleaned query and response data from the `conversation_a` column (selected arbitrarily)

To run the script: 
* Create a huggingface account and download huggingface-CLI locally
* Log in to huggingface-CLI on command line
* Be sure to ```pip install datasets``` if packages `requirements.txt` not installed yet
* Run `python bin/download_data.py`
