How to run?
===========

1. Goto the submission directory: `cd /users/scratch/medical_genomics-2022-04/nadavkraoz`
   
   **First time:** Run `miniconda3/bin/conda init tcsh` and then restart the shell. Start again from step 1, without doing this step.
     
   (This adds miniconda to the `$PATH` env var via `~/.tcshrc`)

2. Activate miniconda environment: `conda activate submission_env`

3. Run `run_task.py` as follows:
   
   * For task 1: `python3 aml_workshop/run_task.py 1 [path_to_beat_rnaseq_data] [inferred_ic50_output_path]`
   
   * For task 2: `python3 aml_workshop/run_task.py 2 [path_to_beat_rnaseq_data] [inferred_ic50_output_path]`
   
   * For task 3: `python3 aml_workshop/run_task.py 3 [correlation_matrix_output_path]`

4. Deactivate miniconda environment: `conda deactivate`


Environment setup (internal use)
================================
1. Install miniconda:

   1. Download https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh

   2. Run `chmod +x Miniconda3-py37_4.11.0-Linux-x86_64.sh`

   3. Run `./Miniconda3-py37_4.11.0-Linux-x86_64.sh` and follow instructions

   4. Run `miniconda3/bin/conda init tcsh`

2. Create miniconda environment: `miniconda3/bin/conda create --name submission_env --file aml_workshop/requirements.txt`
