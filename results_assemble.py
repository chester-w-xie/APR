"""
-------------------------------File info-------------------------
% - File name: results_assemble.py
% - Description:
% -
% -
% - Input:
% - Output:  None
% - Calls: None
% - usage:
% - Version： V1.0
% - Last update: 2022-05-19
%  Copyright (C) PRMI, South China university of technology; 2022
%  ------For Educational and Academic Purposes Only ------
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
------------------------------------------------------------------
"""
import torch
import numpy as np


def show_results_summary(_num_sessions, results_dic, num_trial):
    base_avg_over_sessions = []
    all_avg_novel_over_sessions = []
    pre_avg_novel_over_sessoins = []
    curr_avg_novel_over_sessions = []
    both_avg_over_sessions = []
    #

    print(f'\n =====> Results summary (assemble over {num_trial} trials), (Support set: 5 way 5 shot)')
    print(f'-------------------- Average of class-wise acc (%)--------------------------------')
    print(f'\n Session         ', end=" ")
    for _, n in enumerate(results_dic.keys()):
        print(f'{n}', end="\t")
    print(f'Average', end="\t")

    print(f'\n Base      ', end="\t")
    for _, n in enumerate(results_dic.keys()):
        temp = results_dic[n]['Ave_class_wise_acc_base']
        print(f'{temp * 100:.2f}', end="\t")
        base_avg_over_sessions.append(temp)
    print(f'{np.mean(base_avg_over_sessions) * 100:.2f}', end="\t")

    print(f'\n All Novel       ', end=" ")
    for _, n in enumerate(results_dic.keys()):

        if n == 0:
            print(f'-', end="\t")
        else:
            temp = results_dic[n]['Ave_class_wise_acc_all_novel']
            print(f'{temp * 100:.2f}', end="\t")
            all_avg_novel_over_sessions.append(temp)
    print(f'{np.mean(all_avg_novel_over_sessions) * 100:.2f}', end="\t")

    print(f'\n Previous Novel  ', end=" ")
    for _, n in enumerate(results_dic.keys()):

        if n == 0:
            print(f'-', end="\t")
        else:
            temp = results_dic[n]['Ave_class_wise_acc_previous_novel']
            print(f'{temp * 100:.2f}', end="\t")
            pre_avg_novel_over_sessoins.append(temp)

    print(f'{np.mean(pre_avg_novel_over_sessoins) * 100:.2f}', end="\t")

    print(f'\n Current Novel   ', end=" ")
    for _, n in enumerate(results_dic.keys()):

        if n == 0:
            print(f'-', end="\t")
        else:
            temp = results_dic[n]['Ave_class_wise_acc_current_novel']
            print(f'{temp * 100:.2f}', end="\t")
            curr_avg_novel_over_sessions.append(temp)
    print(f'{np.mean(curr_avg_novel_over_sessions) * 100:.2f}', end="\t")

    print(f'\n Both     ', end="\t")
    for _, n in enumerate(results_dic.keys()):
        temp = results_dic[n]['Ave_acc_of_both']
        print(f'{temp * 100:.2f}', end="\t")
        both_avg_over_sessions.append(temp)
    print(f'{np.mean(both_avg_over_sessions) * 100:.2f}', end="\t")
    print(f'\n --------------------------------------------------------------------------------\n ')

    PD = results_dic[0]['Ave_acc_of_both'] - results_dic[_num_sessions - 1]['Ave_acc_of_both']

    #
    tem = results_dic[0]['Ave_class_wise_acc_base'] - results_dic[_num_sessions - 1]['Ave_class_wise_acc_base']
    AR_overall = tem / results_dic[0]['Ave_class_wise_acc_base']
    AR_overall_avg = AR_overall / (_num_sessions-1)
    MR_overall = 1-AR_overall

    AR_session_list = []
    AR_session_list_temp = []
    for _session in range(1, _num_sessions):
        acc_previous = results_dic[_session - 1]['Ave_class_wise_acc_base']
        acc_current = results_dic[_session]['Ave_class_wise_acc_base']
        AR_session = (acc_previous - acc_current) / acc_previous
        AR_session_list.append(AR_session)
        AR_session_list_temp.append(AR_session*100)
    AR_session_avg = np.mean(AR_session_list)
    MSR_session_avg = 1 - AR_session_avg

    CPI = 0.5 * MSR_session_avg + 0.5 * np.mean(all_avg_novel_over_sessions)

    CPI_V2 = 0.5 * MR_overall + 0.5 * np.mean(all_avg_novel_over_sessions)

    print(' ==> PD: {:.2f} (define by CEC); \n'.format(PD * 100))

    print(' =====> AR_overall: {:.2f}, AR_overall_avg: {:.2f},'
          ' MR_overall: {:.2f}; \n'.format(AR_overall * 100, AR_overall_avg * 100, MR_overall * 100))

    print(
        '  =====> AR_session_avg: {:.2f}, '
        'MSR_session_avg: {:.2f};'.format(AR_session_avg * 100, MSR_session_avg * 100))

    print('  =====> Average of all novel acc over {} incremental sessions: {:.2f};'.format(
        _num_sessions - 1, np.mean(all_avg_novel_over_sessions) * 100))
    print('  =====> CPI: {:.2f} \n'.format(CPI * 100))
    print('  =====> CPI_V2: {:.2f} \n'.format(CPI_V2 * 100))


def get_results_assemble(result_path):
    result_dic = torch.load(result_path)

    new_result_dic = {}
    num_trial = len(result_dic)
    num_sessions = None
    for key in result_dic:
        num_sessions = len(result_dic[key])
    Ave_class_wise_acc_base_list = []
    Ave_class_wise_acc_all_novel_list = []
    Ave_class_wise_acc_previous_novel_list = []
    Ave_class_wise_acc_current_novel_list = []
    Ave_acc_of_both_list = []

    #

    for session in range(num_sessions):
        #
        for trial_key in result_dic.keys():
            result_one_trial = result_dic[trial_key]
            result_one_session = result_one_trial[session]
            # print(f'session {session}, trial {trial_key}， result_one_session: {result_one_session}\n')
            if session == 0:
                Ave_class_wise_acc_base_list.append(result_one_session['Ave_class_wise_acc_base'])
                Ave_acc_of_both_list.append(result_one_session['Ave_acc_of_both'])
            else:
                Ave_class_wise_acc_base_list.append(result_one_session['Ave_class_wise_acc_base'])
                Ave_class_wise_acc_all_novel_list.append(result_one_session['Ave_class_wise_acc_all_novel'])
                Ave_class_wise_acc_previous_novel_list.append(result_one_session['Ave_class_wise_acc_previous_novel'])
                Ave_class_wise_acc_current_novel_list.append(result_one_session['Ave_class_wise_acc_current_novel'])
                Ave_acc_of_both_list.append(result_one_session['Ave_acc_of_both'])
        # print(len(Ave_class_wise_acc_base_list))
        # print(len(Ave_class_wise_acc_all_novel_list))

        # tt = np.mean(Ave_class_wise_acc_all_novel_list)
        # print(tt)
        if session == 0:
            result_one_session_new = {'Ave_class_wise_acc_base': np.mean(Ave_class_wise_acc_base_list),
                                      'Ave_class_wise_acc_all_novel': None,
                                      'Ave_class_wise_acc_previous_novel': None,
                                      'Ave_class_wise_acc_current_novel': None,
                                      'Ave_acc_of_both': np.mean(Ave_acc_of_both_list)
                                      }
        else:
            result_one_session_new = {'Ave_class_wise_acc_base': np.mean(Ave_class_wise_acc_base_list),
                                      'Ave_class_wise_acc_all_novel': np.mean(Ave_class_wise_acc_all_novel_list),
                                      'Ave_class_wise_acc_previous_novel': np.mean(
                                          Ave_class_wise_acc_previous_novel_list),
                                      'Ave_class_wise_acc_current_novel': np.mean(
                                          Ave_class_wise_acc_current_novel_list),
                                      'Ave_acc_of_both': np.mean(Ave_acc_of_both_list)
                                      }
        new_result_dic[session] = result_one_session_new
        #
        Ave_class_wise_acc_base_list = []
        Ave_class_wise_acc_all_novel_list = []
        Ave_class_wise_acc_previous_novel_list = []
        Ave_class_wise_acc_current_novel_list = []
        Ave_acc_of_both_list = []

    show_results_summary(num_sessions, new_result_dic, num_trial)


if __name__ == "__main__":

    result_dir = 'exp/Nsynth-100-FS_Finetune_5way_5shot_2022-06-26-23:02:06/test_results_100_trial.pth'

    get_results_assemble(result_dir)



