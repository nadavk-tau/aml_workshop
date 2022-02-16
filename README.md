How to run?
===========

1. Goto the submission directory: `cd /users/scratch/medical_genomics-2022-04/nadavkraoz`

2. Run `run_task.py` as follows:
   
   * For task 1: `./run.sh python3 aml_workshop/run_task.py 1 [path_to_beat_rnaseq_data] [inferred_ic50_output_path]`
   
   * For task 2: `./run.sh python3 aml_workshop/run_task.py 2 [path_to_beat_rnaseq_data] [inferred_ic50_output_path]`
   
   * For task 3: `./run.sh python3 aml_workshop/run_task.py 3 [correlation_matrix_output_path]`

Environment setup (internal use)
================================
1. Install miniconda:

   1. Download https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh

   2. Run `chmod +x Miniconda3-py37_4.11.0-Linux-x86_64.sh`

   3. Run `./Miniconda3-py37_4.11.0-Linux-x86_64.sh` and follow instructions

   4. Run
   ```
   miniconda3/bin/conda init `basename $SHELL`
   ```

2. Create miniconda environment: `miniconda3/bin/conda create --name submission_env --file aml_workshop/requirements.txt`
