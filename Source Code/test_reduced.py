import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle
from utils import Hps
from utils import DataLoader
from utils import Logger
from solver import Solver
from preprocess.tacotron.norm_utils import spectrogram2wav, get_spectrograms
#from preprocess.tacotron.audio import inv_spectrogram, save_wav
from scipy.io.wavfile import write
#from preprocess.tacotron.mcep import mc2wav
import glob
import os
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hps', help='The path of hyper-parameter set', default='vctk_F_Taiwanese_8.json')
    parser.add_argument('-model', '-m', help='The path of model checkpoint', default='ELE11_and_Tsi-169999.pkl')
    parser.add_argument('-source', '-s', help='The path of source .wav file', default='p223_300.wav')
    parser.add_argument('-target', '-t', help='Target speaker id (integer). Same order as the speaker list when preprocessing (en_speaker_used.txt)',default=2)
    parser.add_argument('-output', '-o', help='output .wav path', default='p223_300_to_Tsi_ALNEW.wav')
    parser.add_argument('-sample_rate', '-sr', default=16000, type=int)
    args = parser.parse_args()

    hps = Hps()
    hps.load(args.hps)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None)
    #solver.load_model('/storage/model/voice_conversion/vctk/clf/norm/wo_tanh/8_speakers_1e-2.pkl-149999')
    solver.load_model(args.model)
    one = True
    if one == True: 
        #spec = np.loadtxt(filename)
        _, spec = get_spectrograms(args.source)
        spec_expand = np.expand_dims(spec, axis=0)
        spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)

        c = Variable(torch.from_numpy(np.array([int(args.target)]))).long().cuda()
        result = solver.test_step(spec_tensor, c, gen=True)
        result = result.squeeze(axis=0).transpose((1, 0))
        print(result.shape)
        wav_data = spectrogram2wav(result)
        write(args.output, rate=args.sample_rate, data=wav_data)
    else:
        directory = '8to8/'
        for filename in glob.glob(os.path.join(directory, '*.wav')):
            _, sub_filename = filename.rsplit('/', maxsplit=1)
            _, spec = get_spectrograms(filename)
            spec_expand = np.expand_dims(spec, axis=0)
            spec_tensor = torch.from_numpy(spec_expand).type(torch.FloatTensor)
            c = Variable(torch.from_numpy(np.array([7]))).long().cuda()
            if spec_tensor.size(1) > 1000:
                result_list = []
                for small_spec_tensor in spec_tensor.split(split_size=400, dim=1):
                    print(small_spec_tensor.size())
                    if small_spec_tensor.size(1) >= 10:
                        result = solver.test_step(small_spec_tensor, c, gen=True)
                        result = result.squeeze(axis=0).transpose((1, 0))
                        result_list.append(result)
                result = np.concatenate(result_list, axis=0)
                print(result.shape)
                wav_data = spectrogram2wav(result)
            else:
                result = solver.test_step(spec_tensor, c, gen=True)
                result = result.squeeze(axis=0).transpose((1, 0))
                print(result.shape)
                wav_data = spectrogram2wav(result)
            write(os.path.join('8to8/convert240/', sub_filename), rate=16000, data=wav_data)