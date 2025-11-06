import sys
import os
import logging

# Добавляем путь до модуля
sys.path.append(os.path.join(os.path.dirname(__file__), 'Three_Letter_Mode'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'iupred2a'))


from _CommonConstants             import _CommonConstants
from _Dull_Sum_tail               import _Dull_Sum_tail
from _c_FourierSmoothed_tail      import _c_FourierSmoothed_tail
from _is_this_aa_here             import _is_this_aa_here
from _Log_occurence_differrence   import _Log_occurence_differrence
from _Log_occurence_differrence_INV   import _Log_occurence_differrence_INV

from _T_statistics_single         import _T_statistics_single
from _T_statistics_single_max     import _T_statistics_single_max
from _T_statistics_single_window  import _T_statistics_single_window
from _DegenerateSequencePredictor import _DegenerateSequencePredictor
from _Iupred2                     import _Iupred2
from _how_close_to_the_end        import _how_close_to_the_end
from _T_statistics_PB_word        import _T_statistics_PB_word
from _Log_Length                  import _Log_Length

def _create_predictors_set_common(filename,common):

  objects=[]

  with open(filename, "r") as file:
    for line in file:
       task_string=line
      # убираем пробельные символы по краям
       stripped = line.strip()
       # если после strip() ничего не осталось — пропускаем
       if not stripped:
           continue
       #print(line)
       words =  line.split()
       if words[0]=='_Log_occurence_differrence':
         objects.append(_Log_occurence_differrence(task_string,common))
       elif words[0]=='_Log_occurence_differrence_INV':
         objects.append(_Log_occurence_differrence(task_string,common))
       elif words[0]=='_Dull_Sum_tail':
         objects.append(_Dull_Sum_tail(task_string,common))
       elif words[0]=='_is_this_aa_here':
         objects.append(_is_this_aa_here(task_string,common))
       elif words[0]=='_c_FourierSmoothed_tail':
         objects.append(_c_FourierSmoothed_tail(task_string,common))
       elif words[0]=='_T_statistics_single':
         objects.append(_T_statistics_single(task_string,common))
       elif words[0]=='_T_statistics_single_window':
         objects.append(_T_statistics_single_window(task_string,common))
       elif words[0]=='_T_statistics_single_max':
         objects.append(_T_statistics_single_max(task_string,common))
       elif words[0]=='_DegenerateSequencePredictor':
         objects.append(_DegenerateSequencePredictor(task_string,common))
       elif words[0]=='_Iupred2':
         objects.append(_Iupred2(task_string,common))
       elif words[0]=='_how_close_to_the_end':
         objects.append(_how_close_to_the_end(task_string,common))
       elif words[0]=='_T_statistics_PB_word':
         objects.append(_T_statistics_PB_word(task_string,common))
       elif words[0]=='_Log_Length':
         objects.append(_Log_Length(task_string,common))

       else:
         print(f'Ignore unknown task: {task_string}')


  return objects

if __name__ == "__main__":

  path_to_frequency_store="../DATA/FrequencyExtrapolation/"
  path_to_aaindex_file="../DATA/aaindex/aaindex.data/aaindex.txt"
  path_to_aaindex_tri_letter_file = "../DATA/aaindex/aaindex.data/aaindex_mutant3.txt"

  common = _CommonConstants(
              path_to_frequency_store,
              path_to_aaindex_file,
              path_to_aaindex_tri_letter_file,
              log_level=logging.DEBUG)


  task_set_file = '/home/milch/HDD/CreatePredictors/BasicPredictors/Assigning_Predictor_Sets/check_PB_words.task'
  objects= _create_predictors_set_common(task_set_file,common)

  print(len(objects))
  for obj in objects:
      print(obj.get_task_string())
