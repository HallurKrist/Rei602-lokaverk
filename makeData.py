###############################################################################################
### authors: Einar Jóhannesson - eij1@hi.is                                                 ###
###          Hallur Kristinn Hallsson - hkh32@hi.is                                         ###
###          Zakia Shafaee Ahmad- zsa1@hi.is                                                ###
###                                                                                         ###
### In order to use this file you have to first install the antropy package                 ###
### https://raphaelvallat.com/antropy/build/html/index.html                                 ###
### you can do that by typing "pip install antropy" in a cell and running that before you   ###
### import this file with "import makeData"                                                 ###
###############################################################################################

import numpy as np
import math
import os
from scipy import signal
from scipy import fft
import antropy as ant
import concurrent.futures
import time
from scipy.fft import rfft, rfftfreq
from pathlib import Path

def segment(file):
  #file: an .atr file
  #output: two lists of lists. d_periods is a list of two value lists representing
  #         the start and end of dummy periods and c_periods is a list of the same
  #         but for contraction periods.
  data = np.loadtxt(file, dtype=str)
  p = 0
  C_ints = []
  D_ints = []
  while(p < data.shape[0]):
    t = []
    t.append(int(data[p,1])-1)
    t.append(int(data[p+1,1])-1)
    if(data[p][6] == 'BD'):
      D_ints.append(t)
    else:
      C_ints.append(t)
    p += 2
  return D_ints, C_ints


def filter(file, cutoff, dataset, d_periods=None, c_periods=None):
  #file: a .dat file containing the frequency info
  #cutoff: a tuple of the lower and upper cutoff frequencies
  #dataset: which dataset we're using. 0 refers to 'TPEHGT', any other value
  #         refers to 'TPEHG'
  #d_periods: default is None. If not None, it must be a list of two value lists
  #         representing the start and end of dummy periods during measurement.
  #c_periods: default is None. If d_periods is None, this must be None as
  #         well. If not None, it must be a list of two-value lists representing
  #         the start and end of contraction periods during measurement.
  #Output: If d_periods is None, returns a nx4 array, with each column representing
  #         each signal filtered through four pole bi-directional bandpass 
  #         butterworth filtering with sample frequency of 20Hz.
  #        If d_periods is not None, returns two lists, each containing four
  #         column arrays of segments from the above, representing the filtered
  #         signals over a specific period of time. First list represents dummy
  #         periods and the second contraction periods.
  data = np.loadtxt(file)
  if(dataset == 0):
    data_ind = [1,3,5]
  else:
    data_ind = [1,5,9]
  data_signals = np.empty([data.shape[0], 4])
  data_signals[:,0:3] = data[:,data_ind]
  data_signals[:,3] = data_signals[:,0] - data_signals[:,1] +data_signals[:,2]

  sig_out = np.empty(data_signals.shape)
  a,b = signal.butter(4, cutoff, btype='bandpass', fs=20)
  for i in range(4):
    sig_out[:,i] = signal.filtfilt(a,b,data_signals[:,i])
  if(d_periods == None):
    assert(c_periods == None)
    return sig_out
  else:
    assert(c_periods != None)
    D_freqs = []
    C_freqs = []
    for d in d_periods:
      D_freqs.append(sig_out[d[0]:d[1]+1,:])
    for c in c_periods:
      C_freqs.append(sig_out[c[0]:c[1]+1,:])
    return D_freqs, C_freqs

def make_path_list(directory_str, text_head=0):
  ##################################################
  # takes in a directory and makes a list with all #
  # paths to .txt files in that directory          #
  #                                                #
  # directory_str : string with the path to the    #
  #                 directory                      #
  #      e.g. '/content/.../dat/tpehgt_d_n002.txt' #
  # text_head : reds .txt files if 0, .hea files   #
  #             else                               #
  # dataset : tpehgt(small) or tpehg(large)        #
  ##################################################
  directory = os.fsencode(directory_str)
  
  list_of_paths = []
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if text_head == 0 and filename.endswith(".txt"): 
      list_of_paths.append(directory_str + filename)
      continue
    if text_head != 0 and filename.endswith(".hea"):
      list_of_paths.append(directory_str + filename)
      continue
    else:
      continue

  return list_of_paths

def make_data_txt_file(X, y):
  #######################################################
  # Takes in data and makes a new file with given data  #
  #######################################################
  # open a (new) file to write
  outF = open("myData.txt", "w")

  for i in range(np.shape(X)[0]):
    row = X[i]
    # write X to output file
    for j in range(48):
      outF.write(str(row[j]))
      outF.write(", ")
    # write y to output file
    outF.write(str(y[i]))
    outF.write("\n")
  # close output file
  outF.close()

def make_data(dat_directory, atr_directory, head_directory, cutoff_list, dataset, period=None):
  ###############################################################################################################################################################
  # dat_directory : string path to directory with .txt files of data                                                                                            #
  # atr_directory : string path to directory with .atr files of dummy/contraction period data or None                                                           #
  # head_directory : string path to directory with .hea files of additional data (used to get y) or None                                                        #
  # cutoff_list : list of 2 value lists where 2 value lists are frequency bands to be filtered by                                                               #
  # dataset : 0 if tpehgt dataset (small dataset) or 1 if tpehg dataset (big dataset). (if 0 then atr_directory not None and if 1 then head_directory not None) #
  # period : "C"/"D" if want to additionally filter by contraction or dummy periods (only possible if atr_directory not None)                                   #
  #                                                                                                                                                             #
  # output : X - [16 sample entropies, 16 median frequencies, 16 peak amplitudes] and y - 0 if term, 1 if preterm and -1 if not pregnant                        #
  ###############################################################################################################################################################
  dat_path_list = sort_paths(make_path_list(dat_directory, 0), dataset)
  head_path_list = None
  atr_path_list = None
  y = []
  if atr_directory != None:
    atr_path_list = sort_paths(make_path_list(atr_directory, 0), dataset)
  if head_directory != None:
    head_path_list = sort_paths(make_path_list(head_directory, 1), dataset)
  
  X = make_data_X(dat_path_list=dat_path_list, cutoff_list=cutoff_list, dataset=dataset, atr_path_list=atr_path_list, period=period)

  if dataset != 0:
    y = make_data_y(head_path_list) # big dataset
  else:
    y = [-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0] #sm dataset
  
  make_data_txt_file(X,y)

  return X, y

def make_data_Xrow(file_, cutoff_list_, dataset_, period=None, d_periods=None, c_periods=None):
  ##########################################################################
  # goes through 1 file and returns the feature data for that file         #
  #                                                                        #
  # file_ :         path to .dat file                                      #
  # cutoff_list_ :  list of cutoffs (tuple)                                # 
  # dataset_ :      which dataset we're using. 0 refers to 'TPEHGT',       #
  #                 any other value refers to 'TPEHG'                      #
  # period :        "C" or "D", refering to contraction- or dummy  period  #
  #                 if None use whole set                                  #
  # d_periods :     list of 2-value lists, refering to dummy periods       #
  # c_periods :     list of 2-value lists, refering to contraction periods #
  #                                                                        #
  # returns :       [ sampen ... median ... peakA ]                        #  
  ##########################################################################
  data_X = [] # to be returned

  sampen_list = None
  median = None
  peak_amplitude = None
  
  if period != None:
    # init all features with period
    sampen_list = file_sampen(file_, cutoff=False, list_of_cutoffs=cutoff_list_, dataset=dataset_, period=period, d_periods=d_periods, c_periods=c_periods) # emb_dim = 3 default
    median, peak_amplitude = file_median_and_peakamplitude(file_, cutoff_list_, dataset_, period, d_periods, c_periods)

  else:
    # init all features with whole file
    sampen_list = file_sampen(file_, cutoff=False, list_of_cutoffs=cutoff_list_, dataset=dataset_) # emb_dim = 3 default
    median, peak_amplitude = file_median_and_peakamplitude(file_, cutoff_list_, dataset_)

  # from martix/array to list
  peak_amplitude_list = []
  median_list = []
  for i in range(np.shape(median)[0]):
    for j in range(np.shape(median)[1]):
      med = np.array(median)[i][j]
      peak = np.array(peak_amplitude)[i][j]
      median_list.append(med)
      peak_amplitude_list.append(peak)
  
  for i in range(len(sampen_list)):
    data_X.append(sampen_list[i])

  for i in range(len(median_list)):
    data_X.append(median_list[i])

  for i in range(len(peak_amplitude_list)):
    data_X.append(peak_amplitude_list[i])

  return data_X

def make_data_X(dat_path_list, cutoff_list, dataset, atr_path_list, period=None):
  ##########################################################################
  # goes through list of files and returns the feature data for those files#
  #                                                                        #
  # dat_path_list : list of paths to .txt files                            #
  # cutoff_list :   list of cutoffs (tuple)                                # 
  # dataset :       which dataset we're using. 0 refers to 'TPEHGT',       #
  #                 any other value refers to 'TPEHG'                      #
  # atr_path_list : list of paths to .atr files                            #
  # period :        "C" or "D", refering to contraction- or dummy  period  #
  #                 if None use whole set                                  #
  #                                                                        #
  # returns :       [ sampen ... median ... peakA ]dat_path_list[0]        #  
  #                 [ sampen ... median ... peakA ]dat_path_list[1]        # 
  #                 [ sampen ... median ... peakA ]dat_path_list[2]        # 
  #                               ...                                      #
  #                 [ sampen ... median ... peakA ]dat_path_list[n]        # 
  ##########################################################################
  X = np.empty((len(dat_path_list), 48)) # 48 features

  for i in range(len(dat_path_list)):
    path = dat_path_list[i]

    if period != None:
      d_periods, c_periods = segment(atr_path_list[i])
      if period == "D":
        X[i] = make_data_Xrow(path, cutoff_list, dataset, period, d_periods, c_periods)
      else:
        X[i] = make_data_Xrow(path, cutoff_list, dataset, period, d_periods, c_periods)
    else:
      X[i] = make_data_Xrow(path, cutoff_list, dataset)
  
  return X

def calc_sampen(signal_, emb_):
  # Calculates the sampel entropy for a given time series signal_ and embedded dimension emb_
  # using the antropy package
  return ant.sample_entropy(signal_, order=emb_)

def file_sampen(file_, emb_dim=3, cutoff=True, list_of_cutoffs=None, dataset=0, period=None, d_periods=None, c_periods=None):
  # Calculates the sample entropy for a given file with embedded dimension emb_dim 
  # and filters the frequencies into categories by list_of_cutoffs if cutoff=False.
  #
  # file -> a filepath to a .dat file
  # emb_dim -> the embedding dimension to use in calculating the sample entropy, default = 3
  # cutoff -> True if using default cutoffs [0.08, 1][1, 2.2][2.2, 3.5][3.5, 5], False if using list_of_cutoffs
  # list_of_cutoffs -> has to be a list of cutoffs if cutoff != True
  #                    e.g. [[x1, y1],[x2, y2],...]
  # dataset -> which dataset we're using. 0 refers to 'TPEHGT', any other value refers to 'TPEHG'     
  # period :       "D" or "C" lets know what period to use, if None then whole file                    
  # d_periods :    list of 2-value lists, refering to dummy periods                                    
  # c_periods :    list of 2-value lists, refering to contraction periods                              
  #
  # output -> returns a list of sample entropies of length 4*len(list_of_cutoffs) if cutoff != True
  #           else of length 4*4=16.
  #           where you have the sample entropy for signals 1-4 for each cutoff

  cutoff_list = []
  if cutoff == True:
    cutoff_list = [[0.08, 1],[1, 2.2],[2.2, 3.5],[3.5, 5]]
  else:
    cutoff_list = list_of_cutoffs

  sample_entropies = np.empty((4*len(cutoff_list),))

  filtered_file=None
  for j, cut in enumerate(cutoff_list):
    if period != None:
      d,c = filter(file_, cut, dataset, d_periods, c_periods)
      if period == "D":
        filtered_file = d
      else:
        filtered_file = c
    else:
      filtered_file = filter(file_, cut, dataset)
    for i in range(4):
      sample_entropies[j*4+i] = calc_sampen(filtered_file[:,i], emb_dim)
    
  return sample_entropies

###############################################################################
### Ran this function on embedding dimensions 4, 8 and 16 which returned 4. ###
### Then we ran it for 2, 4 and 6 which returned 2.                         ###
### So then we ran it for 2, 3 and 4 (because 2 is the lowest possible      ### 
### embedding dimension for the antropy package) which returned 3.          ###
### So we conclude that 3 is the optimal embedding dimension for this       ###
### particular project                                                      ###
###############################################################################
def find_optimal_sampen_params(embedding_dimension):
  # finds the embedding dimension that gives the most variance between preterm and term
  # pregnansies out of a list of 3 possible embedding dimensions e.g. [2, 4, 6]
  #
  # embedding_dimension has to be a list/array of 3 possible embedding dimensions of 2 or above

  datpath = '/content/gdrive/MyDrive/gerfig_lokavrk/dat/'
  dat_name = 'tpehgt_d_'

  dat_type = ['p', 't']
  dat_p_num = ['001','002','003','004','005','006','007','008','009','010','011','012','013']
  dat_t_num = ['001','002','003','004','005','006','007','008','009','010','011','012','013']
  dat_num = [dat_p_num, dat_t_num]

  dat_ext = '.txt'

  cutoff1 = [0.08, 1]
  cutoff2 = [1, 2.2]
  cutoff3 = [2.2, 3.5]
  cutoff4 = [3.5, 5]

  emb_dim = embedding_dimension

  #calculate enthropy for preterm 
  preterm_entropy = []
  for i in range(len(dat_num[0])):
    file = datpath + dat_name + dat_type[0] + dat_num[0][i] + dat_ext

    print("starting on file: ", end="")
    print(file)

    sig_out_1 = filter(file, cutoff1, 0)
    sig_out_2 = filter(file, cutoff2, 0)
    sig_out_3 = filter(file, cutoff3, 0)
    sig_out_4 = filter(file, cutoff4, 0)

    outs = [sig_out_1, sig_out_2, sig_out_3, sig_out_4]

    with concurrent.futures.ThreadPoolExecutor() as executor:
      start = time.perf_counter()
      for v in range(len(emb_dim)):
        for out in outs:
          for u in range(4):
            preterm_entropy.append(executor.submit(calc_sampen, out[:,u], emb_dim[v]))

    print(f'Duration: {time.perf_counter() - start}')
          
  #calculate enthropy for term 
  term_entropy = []
  for i in range(len(dat_num[1])):
    file = datpath + dat_name + dat_type[1] + dat_num[1][i] + dat_ext

    print("starting on file: ", end="")
    print(file)

    sig_out_1 = filter(file, cutoff1, 0)
    sig_out_2 = filter(file, cutoff2, 0)
    sig_out_3 = filter(file, cutoff3, 0)
    sig_out_4 = filter(file, cutoff4, 0)

    outs = [sig_out_1, sig_out_2, sig_out_3, sig_out_4]

    with concurrent.futures.ThreadPoolExecutor() as executor:
      start = time.perf_counter()
      for v in range(len(emb_dim)):
        for out in outs:
          for u in range(4):
            term_entropy.append(executor.submit(calc_sampen, out[:,u], emb_dim[v]))

    print(f'Duration: {time.perf_counter() - start}')

  ####### THIS IS A STATUS UPDATE #######
  # preterm_entropy and term_entropy are now arrays where every 48 enteries are from a file
  # e.g. preterm_entropy[:48] is all from the first preterm file.
  # And every 16 enteries in the 48 are the entropy calculated with a different embedding dimension
  # And lastly those 16 enteries are the 4 signal outputs with their 4 columns
  # e.g. the first 4 enteries in the 16 are the 4 columns of the first signal
  # 
  # tldr: [-------------------------------------------------file1 (48 enteries)--------------------------------------------------------------------------------------, file2 (48 enteries), ...]
  #       [--------------------------------emb_dim1 (16 enteries)----------------------------------------------, emb_dim2 (16 enteries), emb_dim3 (16 enteries), ... , emb_dim1 (16 enteries),...]
  #       [----signal1 (4 enteries)----, signal2 (4 enteries), signal3 (4 enteries), signal4 (4 enteries), ... ,signal1 (4 enteries),...]
  #       [---col1, col2, col3, col4---, col1, ...]


  ### GOAL ###
  # to make collapse preterm_entropy and term_entropy into 48 entry lists
  # then compare them and choose the embedding dimension that gives the biggest difference between them

  collapsed_preterm = np.empty((48,))
  collapsed_term = np.empty((48,))
  for j in range(48):
    tmp_pre = preterm_entropy[j::48]
    tmp_term = term_entropy[j::48]
    for i in range(len(tmp_pre)):
      tmp_pre[i] = tmp_pre[i].result() 
    for i in range(len(tmp_term)): 
      tmp_term[i] = tmp_term[i].result() 

    collapsed_preterm[j] = np.mean(tmp_pre)
    collapsed_term[j] = np.mean(tmp_term)

  diff = np.empty((3,))
  for i in range(3):
    emb_diff = np.empty((16,))  
    for j in range(16):
      emb_diff[j] = np.abs(collapsed_preterm[i*16+j] - collapsed_term[i*16+j])
    diff[i] = np.sum(emb_diff)

  index_most_diff = np.argmax(diff)
  
  # returns the embedding dimension that gives the most difference between preterm and term pregnancies
  return emb_dim[index_most_diff]

def Median(a):
  ################################################################
  # calculates medain frequency from a signal                    #
  #                                                              #
  # a : a signal (time series)                                   ##
  ################################################################
  N = len(a)
  TMPList = np.sort(a)

  if (N % 2) == 0:
    median = (TMPList[int(N/2)] + TMPList[int(N/2) + 1])/2
  else:
    median = TMPList[math.ceil(N/2)]
  return median

def Peak_amplitude(a, cutoff):
  ################################################################
  # calculates peak amplitude from a signal and a frequency band #
  #                                                              #
  # a : a signal (time series)                                   #
  # cutoff : frequency band                                      #
  ################################################################
  fs = 20    #Hertz
  yf = rfft(a)
  #power spectra
  ps = 2*(np.abs(yf)**2.0)
  M = len(ps)
  xf = rfftfreq(M, 1/fs)
  idx = np.logical_and(xf >= cutoff[0], xf <= cutoff[1])
  cutoff_idx = np.where(idx)[0]
  #peak amplitude
  pa = max(ps[cutoff_idx]/max(ps))
  return pa

def file_median_and_peakamplitude(file_, cutoff_list_, dataset_, period=None, d_periods=None, c_periods=None):
  ######################################################################################################
  # function to find median requency and peak amplitude from file, file_                                                  #
  #                                                                                                    #
  # file_ :        the file path to find median frequency off                                          #
  # cutoff_list_ : list of cutoffs that the file is filtered with                                      #
  # dataset_ :     which dataset we're using. 0 refers to 'TPEHGT', any other value refers to 'TPEHG'  #
  # period :       "D" or "C" lets know what period to use, if None then whole file                    #
  # d_periods :    list of 2-value lists, refering to dummy periods                                    #
  # c_periods :    list of 2-value lists, refering to contraction periods                              #
  #                                                                                                    #
  # Returns :      2 ndarrays of shape (len(cutoff_list_), 4)                                           #
  #                where each row is a cutoff and each column is the signal 1-4 of that cutoff         #
  ######################################################################################################
  signals_ = []
  for i in range(len(cutoff_list_)):
    if period != None:
      d,c = filter(file_, cutoff_list_[i], dataset_, d_periods, c_periods)
      if period == "D":
        signals_.append(d)
      else:
        signals_.append(c)
    else:
      signals_.append(filter(file_, cutoff_list_[i], dataset_))

  median = np.empty((np.shape(signals_)[0], 4))
  peak_amplitude = np.empty((np.shape(signals_)[0], 4))
  for i in range(len(signals_)):
    for j in range(4):
      f, pxx = signal.welch(signals_[i][:,j])
      median[i,j] = Median(np.array(pxx))
      peak_amplitude[i,j] = Peak_amplitude(signals_[i][:,j], cutoff_list_[i])
  
  return median, peak_amplitude

def getStats(path):
  # gets some statistics from .hea file in path
  f=open(path)
  lines=f.readlines()
  lin_gst = lines[15].split(' ')[5]
  lin_rtm = lines[16].split(' ')[5]
  lin_age = lines[17].split(' ')[5]
  res = []
  #if(lin_gst=='None\n'):
  #    res.append(-1);
  #else:
  res.append(float(lin_gst))
  #if(lin_rtm=='None\n'):
  #    res.append(-1);
  #else:
  res.append(float(lin_rtm))
  if(lin_age=='None\n'):
      res.append(-1)
  else:
      res.append(float(lin_age))
  return res

def extract_nrs_from_dat_paths(path_list, name):
  ###################################################################################
  # path_list : list of strings where each string is a path to a file               #
  # name : the kind of file. tpehg or tpehgt                                        #
  #                                                                                 #
  # output : nrs_list = list of numbers where each number is the number of the file #
  #          e.g. path_list[i] = ..../tpehg1000.txt -> nrs_list[i] = 1000           #
  ###################################################################################
  nrs_list = []
  index = -1
  for i in range(len(path_list)):
    split = path_list[i].split(sep="/")
    if index == -1:
      for j in range(len(split)):
        if split[j][0:4] == name:
          index = j
    nr_txt = split[index][5:]
    nrs_list.append(int(nr_txt.split(sep=".")[0]))
  return nrs_list

def sort_paths(path_list, dataset):
  # sorts the paths in path_list based on the number of the tpehg/tpehgt file
  name = ""
  sorted_ = []
  if dataset == 0:
    name = "tpehg"
    index_sort = np.argsort(extract_nrs_from_dat_paths(path_list, name))
    for i in range(index_sort):
      sorted_[i] = path_list[index_sort[i]]
  else:
    name = "tpehgt"
    sorted_ = sorted(path_list)

  return sorted_

def make_data_y(head_path_list):
  # makes the y data from the .hea files, referecned from
  # the paths in head_path_list

  # [file number, Gestation, Rectime, Age]
  Data = np.zeros([len(head_path_list),4])

  x = 0
  for i in range(len(head_path_list)):
    path = head_path_list[i]
    if(Path(path).is_file()):
      Data[x,0] = i
      Data[x,1:] = getStats(path)
      x = x+1
    
  # 1 = preterm, 0 = term
  y = np.zeros(len(head_path_list))
  for i, d in enumerate(Data):
    if(d[1]<37):
        y[i] = 1
    else:
        y[i] = 0
        
  return y

def make_DC_period_files(dat_directory, atr_directory, cutoff_list):
  # makes 2 files dummyData.txt and contractionData.txt where in 
  # dummyData every line is all features for 1 dummy period and vice versa for contractionData
  dat_path_list = sorted(make_path_list(dat_directory))
  atr_path_list = sorted(make_path_list(atr_directory))

  outD = open("dummyData.txt", "w")
  outC = open("contractionData.txt", "w")

  for i in range(len(dat_path_list)):

    y_value = None

    if i < 5:
      y_value = -1
    if i >= 5 and i < 18:
      y_value = 1
    if i >= 18:
      y_value = 0

    d_seg, c_seg = segment(atr_path_list[i])

    all_d_sampen = None
    all_d_median = None
    all_d_PA = None

    all_c_sampen = None
    all_c_median = None
    all_c_PA = None

    for j in range(len(cutoff_list)):
      d,c = filter(dat_path_list[i], cutoff_list[j], 0, d_seg, c_seg)

      all_d_sampen = np.empty((np.shape(d)[0], 16))
      all_d_median = np.empty((np.shape(d)[0], 16))
      all_d_PA = np.empty((np.shape(d)[0], 16))

      all_c_sampen = np.empty((np.shape(d)[0], 16))
      all_c_median = np.empty((np.shape(d)[0], 16))
      all_c_PA = np.empty((np.shape(d)[0], 16))

      for q in range(np.shape(d)[0]):
        for p in range(4):
          signal = np.array(d[q])[:,p]
          sampen = calc_sampen(signal, 3)
          median = Median(signal)
          PA = Peak_amplitude(signal, cutoff_list[j])

          all_d_sampen[q][j*p] = sampen
          all_d_median[q][j*p] = median
          all_d_PA[q][j*p] = PA

      for q in range(np.shape(c)[0]):
        for p in range(4):
          signal = np.array(c[q])[:,p]
          sampen = calc_sampen(signal, 3)
          median = Median(signal)
          PA = Peak_amplitude(signal, cutoff_list[j])

          all_c_sampen[q][j*p] = sampen
          all_c_median[q][j*p] = median
          all_c_PA[q][j*p] = PA

    for p in range(np.shape(all_d_sampen)[0]):
      for samp in all_d_sampen[p]:
        outD.write(str(samp))
        outD.write(", ")
      for med in all_d_median[p]:
        outD.write(str(med))
        outD.write(", ")
      for p in all_d_PA[p]:
        outD.write(str(p))
        outD.write(", ")

      outD.write(str(y_value))
      outD.write("\n")  

    for p in range(np.shape(all_c_sampen)[0]):
      for samp in all_c_sampen[p]:
        outC.write(str(samp))
        outC.write(", ")
      for med in all_c_median[p]:
        outC.write(str(med))
        outC.write(", ")
      for p in all_c_PA[p]:
        outC.write(str(p))
        outC.write(", ")

      outC.write(str(y_value))
      outC.write("\n")

  outD.close()
  outC.close()