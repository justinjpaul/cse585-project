#!/bin/bash
#SBATCH --job-name=cse585_model_train
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=engin1
#SBATCH --partition=gpu
if [ ! -d "venv" ]; then
    # Create the virtual environment
    python3 -m venv venv
    echo "Virtual environment created."
else
    source venv/bin/activate
    echo "Virtual environment already exists."
fi

source venv/bin/activate
pip install -r requirements.txt


python response_prediction_model/model.py