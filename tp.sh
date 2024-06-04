#!/bin/bash
#SBATCH -N 3
#SBATCH --ntasks-per-node=20
#SBATCH --time=24:00:00
#SBATCH --job-name=Test
#SBATCH --error=%J.err_
#SBATCH --output=%J.out_
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=108119076@nitt.edu


module load DL/DL-CondaPy/3.7
source /home/apps/spack/share/spack/setup-env.sh
spack load py-tensorflow@2.10.1
module load cuda/10.2
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
python tp.py