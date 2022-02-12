import argparse
import pickle
import pandas as pd
import numpy as np

from config import path_consts
from enum import Enum


class Tasks(Enum):
    task1 = '1'
    task2 = '2'
    task3 = '3'

MODELS_NAME_BY_TASK= {
    Tasks.task1: r'task1.model',
    Tasks.task2: r'task2.model'
}

DRUGS_LIST = ['A-674563', 'Afatinib (BIBW-2992)', 'Alisertib (MLN8237)',
       'Axitinib (AG-013736)', 'AZD1480', 'Barasertib (AZD1152-HQPA)',
       'BEZ235', 'BMS-345541', 'Bortezomib (Velcade)', 'Bosutinib (SKI-606)',
       'Canertinib (CI-1033)', 'Cediranib (AZD2171)', 'CHIR-99021',
       'CI-1040 (PD184352)', 'Crenolanib', 'Crizotinib (PF-2341066)', 'CYT387',
       'Dasatinib', 'Doramapimod (BIRB 796)', 'Dovitinib (CHIR-258)',
       'Erlotinib', 'Flavopiridol', 'Foretinib (XL880)', 'GDC-0879',
       'Gefitinib', 'GSK-1838705A', 'GSK-1904529A', 'GSK690693', 'Idelalisib',
       'Imatinib', 'INK-128', 'JAK Inhibitor I', 'JNJ-38877605', 'JNJ-7706621',
       'KI20227', 'KU-55933', 'KW-2449', 'Lapatinib', 'Linifanib (ABT-869)',
       'LY-333531', 'Masitinib (AB-1010)', 'Midostaurin', 'MLN120B', 'MLN8054',
       'Motesanib (AMG-706)', 'Neratinib (HKI-272)', 'Nilotinib', 'NVP-TAE684',
       'Pazopanib (GW786034)', 'PD173955', 'Pelitinib (EKB-569)', 'PHA-665752',
       'PI-103', 'Ponatinib (AP24534)', 'PP242', 'PRT062607',
       'Quizartinib (AC220)', 'RAF265 (CHIR-265)', 'Rapamycin',
       'Regorafenib (BAY 73-4506)', 'Roscovitine (CYC-202)',
       'Ruxolitinib (INCB018424)', 'SB-431542', 'Selumetinib (AZD6244)',
       'SGX-523', 'SNS-032 (BMS-387032)', 'Sorafenib', 'STO609', 'Sunitinib',
       'TG100-115', 'Tofacitinib (CP-690550)', 'Tozasertib (VX-680)',
       'Trametinib (GSK1120212)', 'Vandetanib (ZD6474)', 'Vargetef',
       'Vatalanib (PTK787)', 'Vismodegib (GDC-0449)', 'VX-745', 'YM-155']

def _parse_input():
    parser = argparse.ArgumentParser(description='Run tasks prediction')
    parser.add_argument('--task-id', '-tid', type=Tasks, help='The task id to run')
    parser.add_argument('--input-file', '-i', type=str, nargs='?', help='Input file')
    parser.add_argument('--output-file', '-o', type=str, help='Output file')

    parsed_args = parser.parse_args()
    if (parsed_args.task_id == Tasks.task3) and (parsed_args.input_file is not None):
        raise "Task3 it running without input file"

    return parsed_args

def _load_pickle_model(task):
    model_path = path_consts.FINAL_TRAINNED_MDOELS_PATH / MODELS_NAME_BY_TASK[task]
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

def _transform_input_data(input_data):
    input_data = input_data.T
    input_data = input_data.fillna(input_data.mean())

    return np.log2(input_data + 1)

def _transfome_output_results(results, patient_names):
    results = pd.DataFrame(results)
    results.columns = DRUGS_LIST
    results.set_index(patient_names, inplace=True)

    return results.T

def _run_model_tasks(parsed_args):
    trainniend_model = _load_pickle_model(parsed_args.task_id)
    input_data = _transform_input_data(pd.read_csv(parsed_args.input_file, sep='\t'))

    predictions = trainniend_model.predict(input_data)
    nimrod_output = _transfome_output_results(predictions, input_data.index)
    nimrod_output.to_csv(parsed_args.output_file, sep='\t')

def _run_task3(parsed_args):
    print("Task3")

_RUNNING_TASKS = {
    Tasks.task1: _run_model_tasks,
    Tasks.task2: _run_model_tasks,
    Tasks.task3: _run_task3,
}

def main():
    parsed_args = _parse_input()
    task = parsed_args.task_id

    _RUNNING_TASKS[task](parsed_args)


if __name__ == '__main__':
    main()
