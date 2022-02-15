import argparse
import pandas as pd
import numpy as np

from joblib import load
from config import path_consts
from enum import Enum


class Tasks(Enum):
    task1 = '1'
    task2 = '2'
    task3 = '3'


MODELS_NAME_BY_TASK= {
    '1': r'task1.model',
    '2': r'task2.model'
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


def _parse_args():
    parser = argparse.ArgumentParser(description='Run tasks')
    subparsers = parser.add_subparsers(dest='task_id')
    subparsers.required = True

    task1_parser = subparsers.add_parser('1')
    task1_parser.add_argument('input_file', type=str, help='Input file')
    task1_parser.add_argument('output_file', type=str, help='Output file')
    task1_parser.set_defaults(func=_run_model_tasks)

    task2_parser = subparsers.add_parser('2')
    task2_parser.add_argument('input_file', type=str, help='Input file')
    task2_parser.add_argument('output_file', type=str, help='Output file')
    task2_parser.set_defaults(func=_run_model_tasks)

    task3_parser = subparsers.add_parser('3')
    task3_parser.add_argument('output_file', type=str, help='Output file')
    task3_parser.set_defaults(func=_run_task3)

    return parser.parse_args()


def _load_model(task):
    model_path = path_consts.FINAL_TRAINNED_MDOELS_PATH / MODELS_NAME_BY_TASK[task]
    with open(model_path, 'rb') as model_file:
        return load(model_file)


def _transform_input_data(input_data):
    input_data = input_data.T
    input_data = input_data.fillna(input_data.mean())

    return np.log2(input_data + 1)


def _transform_output_results(results, patient_names):
    results = pd.DataFrame(results)
    results.columns = DRUGS_LIST
    results.set_index(patient_names, inplace=True)

    return results.T


def _run_model_tasks(parsed_args):
    print(f"Running task {parsed_args.task_id}:")
    print("> Loading model...")
    trained_model = _load_model(parsed_args.task_id)
    print(f"> Loading input file '{parsed_args.input_file}'...")
    input_data = _transform_input_data(pd.read_csv(parsed_args.input_file, sep='\t'))
    print("> Inferring...")
    predictions = trained_model.predict(input_data)
    print(f"> Writing output to '{parsed_args.output_file}'...")
    transformed_output = _transform_output_results(predictions, input_data.index)
    transformed_output.to_csv(parsed_args.output_file, sep='\t')
    print("Done.")


def _run_task3(parsed_args):
    print("Task3")


def main():
    parsed_args = _parse_args()
    parsed_args.func(parsed_args)


if __name__ == '__main__':
    main()
