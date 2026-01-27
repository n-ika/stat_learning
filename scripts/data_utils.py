import torch; torch.manual_seed(108)
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa

def normalize_data(data,min_data,max_data):
    data_normed = (data - min_data) / (max_data - min_data + 1e-6)
    return(data_normed)

def get_syll_vec(model_type,root_raw_stims,exp,n_freqchannels):
    if exp==1:
        sylls = ['pa','bi','ku','ti','bu','do','go','la','tu','da','ro','pi']
    elif exp==2:
        sylls = ['gu','lo','ba','bo','ga','ki','ge','bi','tu','pi','du','go']
    if model_type=='onehot':
        syll_vec=get_onehot_vecs(sylls)
    elif model_type=='phon':
        syll_vec=get_phon_vecs(sylls)
    elif model_type.startswith('acoustic'):
        syll_vec=get_acoustic_vecs(root_raw_stims,sylls,n_freqchannels)
    return(syll_vec)

def get_onehot_vecs(sylls):
    syll_vec = {}
    onehot = F.one_hot(torch.arange(0, 12).view(12,1,1), num_classes=12)
    # randomize order
    sylls = np.random.choice(sylls, size=12, replace=False)
    for i in range(12):
        syll_vec[sylls[i]] = onehot[i]
    syll_vec['null'] = torch.zeros((1,1,12))
    return(syll_vec)

def get_phon_vecs(sylls,stims_file=None):
    if stims_file is None:
        stims_file = 'data/phonological_mapping.txt'
    encoding_unigram = pd.read_csv(stims_file,header=None,sep='\s+')
    # make encodings into a dictionary
    encodings = {}
    for i in range(encoding_unigram.shape[0]):
        encodings[encoding_unigram.loc[i,0]] = encoding_unigram.loc[i,1:].values
    syll_vec = {}
    for itm in sylls:
        if itm not in syll_vec.keys():
            syll_vec[itm] = np.reshape(np.vstack([encodings[itm[0]],encodings[itm[1]]]).astype('float32'),(1,1,encodings[itm[0]].shape[0]*2))
    syll_vec['null'] = np.zeros([1,1,encodings[itm[0]].shape[-1]*2])
    return(syll_vec)

def get_acoustic_vecs(root_stims,sylls_ex,n_freqchannels):
    # tgt_sr = 16000
    n_fft=256
    syll_vec = {}
    for syll in sylls_ex:
        y, sr = librosa.load(root_stims+syll+'.wav', sr=None)
        print('sampling rate ', sr)
        # audio = librosa.resample(y, orig_sr=sr, target_sr=tgt_sr)
        mel_specgram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft//4, n_mels=n_freqchannels, window='hann')
        # TODO: log the mel spectrogram
        syll_vec[syll] = np.reshape(mel_specgram[np.newaxis, :], (1, 1, -1))
    syll_vec['null'] = np.reshape(np.zeros(mel_specgram[np.newaxis, :].shape), (1, 1, -1))
    return(syll_vec)

def append_data(syll_vec,data,label_in1,label_out,condition=None,label_in2=None):
    data['data_in'].append(syll_vec[label_in1])
    if label_in2 is not None:
        data['data_in'].append(syll_vec[label_in2])
        data['labels_in'].append('-'.join((label_in1,label_in2)))
    else:
        data['labels_in'].append(label_in1)
    data['data_out'].append(syll_vec[label_out])
    data['labels_out'].append(label_out)
    if condition is not None:
        data['conditions'].append(condition)
    return(data)


def create_train_data(transcript_df,syll_vec,stim_type):
    train = {'data_in':[],'data_out':[],
            'labels_in':[],'labels_out':[],
            'conditions':[]}
    for i in range(len(transcript_df)):
        row = transcript_df.iloc[i]
        word=row['stimulus'].split('-')
        # condition = 1 if row['stimulus'] in train_items.keys() else 0
        if i < len(transcript_df)-1:
            next_row = transcript_df.iloc[i+1]
            next_word=next_row['stimulus'].split('-')
            # condition_next = 1 if next_row['stimulus'] in train_items.keys() else 0
        if stim_type=='unigram':
            N = 1
            train = append_data(syll_vec,train, word[0], word[1], 1, None)
            train = append_data(syll_vec,train, word[1], word[2], 1, None)
            if i < len(transcript_df)-1:
                train = append_data(syll_vec,train, word[2], next_word[0], 0, None)
        else: 
            N = 2
            if stim_type=='bigram':
                train = append_data(syll_vec, train, word[0], word[2], 1, word[1])
                train = append_data(syll_vec, train, word[1], next_word[0], 0, word[2])
                if i < len(transcript_df)-1:
                    train = append_data(syll_vec, train, word[2], next_word[1], 0, next_word[0])
            elif stim_type=='zerovec-bigram':
                train = append_data(syll_vec, train, 'null', word[1], 1, word[0])
                train = append_data(syll_vec, train, word[0], word[2], 1, word[1])
    train_in = np.reshape(np.stack(train['data_in'], axis=0),(-1,N,syll_vec['pi'].shape[2]))
    train_out = np.reshape(np.stack(train['data_out'], axis=0),(-1,1,syll_vec['pi'].shape[2]))
    labels_in=train['labels_in']
    labels_out=train['labels_out']
    conditions=train['conditions']
    return(train_in, labels_in, train_out, labels_out, conditions)

def create_test_data(test_items,syll_vec,stim_type):
    test = {'data_in':[],'data_out':[],
            'labels_in':[],'labels_out':[],
            'conditions':[]}
    for test_itm in test_items.keys():
        word=test_itm.split('-')
        condition=test_items[test_itm]
        if stim_type=='zerovec-bigram':
            N = 2
            test = append_data(syll_vec, test, 'null', word[1], condition, word[0])
            test = append_data(syll_vec, test, word[0], word[2], condition, word[1])
        elif stim_type=='bigram':
            N = 2
            test = append_data(syll_vec, test, word[0], word[2], condition, word[1])
        elif stim_type=='unigram':
            N = 1
            test = append_data(syll_vec, test, word[0], word[1], condition, None)
            test = append_data(syll_vec, test, word[1], word[2], condition, None)
    test_in = np.reshape(np.stack(test['data_in'], axis=0),(-1,N,syll_vec['pi'].shape[2]))
    test_out = np.reshape(np.stack(test['data_out'], axis=0),(-1,1,syll_vec['pi'].shape[2]))
    labels_in=test['labels_in']
    labels_out=test['labels_out']
    conditions=test['conditions']
    return(test_in, labels_in, test_out, labels_out, conditions)

def reshape_syll_vec(syll_vec_):
    syll_vec = {}
    for key in syll_vec_.keys():
        syll_vec[key] = torch.tensor(syll_vec_[key].astype(np.float32).reshape(1, 1, -1))
    return(syll_vec)

def save_data(name,inputs,labels,conditions,out_dir,model_type):
    if conditions is None:
        torch.save({
            'inputs': inputs,  
            'labels': labels   
        }, out_dir+f'{model_type}_{name}.pt')
    else:
        torch.save({
            'inputs': inputs,  
            'labels': labels,
            'conditions': conditions
        }, out_dir+f'{model_type}_{name}.pt')
    return()