#!/bin/bash
#SBATCH --job-name=cse585_model_train
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=engin1
#SBATCH --partition=gpu
#SBATCH --get-user-env
#SBATCH --mem=5g

if [ ! -d "venv" ]; then
    # Create the virtual environment
    python3 -m venv venv
    echo "Virtual environment created."
else
    source venv/bin/activate
    echo "Virtual environment already exists."
fi

source venv/bin/activate
module load python
pip install -r requirements.txt


python response_prediction_model/model.py