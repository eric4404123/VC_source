import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from solver import Solver
from norm_utils import spectrogram2wav, get_spectrograms
from scipy.io.wavfile import write
import glob
import os

if __name__ == '__main__':
    multi = 's'
    hps = Hps()
    hps.load('vctk_10_0315.json')
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)

    solver.load_model('model_Tsi_0315.pkd')

    #單檔轉換
    if multi == 's':
        filename = 'Combine003_Sync061.wav' 

        _, spec = get_spectrograms(filename)
        spec_expand = np.expand_dims(spec, axis=0)
        spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)

        c = Variable(torch.from_numpy(np.array([3]))).cuda()

        result = solver.test_step(spec_tensor, c, gen=True)
            
        result = result.squeeze(axis=0).transpose((1, 0))
        wav_data = spectrogram2wav(result)
        write('p220_109_to_Tsi', rate=16000, data=wav_data)

    #整批資料夾轉換
    else:
        directory = 'voice_conversion/preprocess/VCTK-Corpus/wav48/p227'
        for filename in glob.glob(os.path.join(directory, '*.wav')):
            _, sub_filename = filename.rsplit('/', maxsplit=1)
            _, spec = get_spectrograms(filename)
            spec_expand = np.expand_dims(spec, axis=0)
            spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
            c = Variable(torch.from_numpy(np.array([3]))).cuda()
            if spec_tensor.size(1) > 1000:
                result_list = []
                for small_spec_tensor in spec_tensor.split(split_size=400, dim=1):
                    print(small_spec_tensor.size())
                    if small_spec_tensor.size(1) >= 10:
                        result = solver.test_step(small_spec_tensor, c, gen=False)
                        result = result.squeeze(axis=0).transpose((1, 0))
                        result_list.append(result)
                result = np.concatenate(result_list, axis=0)
                print(result.shape)
                wav_data = spectrogram2wav(result)
            else:
                result = solver.test_step(spec_tensor, c, gen=False)
                result = result.squeeze(axis=0).transpose((1, 0))
                print(result.shape)
                wav_data = spectrogram2wav(result)
            write(os.path.join('/media/msplab/d8ad5387-7e7b-49bd-9643-b97070e119b7/Multi_target_Voice_Conversion/voice_conversion/0419src/p227toTsi', sub_filename), rate=16000, data=wav_data)