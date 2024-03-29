import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
from norm_utils import get_spectrograms 

def read_speaker_info(path='MixtureCorpus/speaker-info.txt'):
    #讀出speaker資訊
    accent2speaker = defaultdict(lambda: [])
    with open(path) as f:
        splited_lines = [line.strip().split() for line in f][1:]
        speakers = [line[0] for line in splited_lines]
        regions = [line[3] for line in splited_lines]
        for speaker, region in zip(speakers, regions):
            accent2speaker[region].append(speaker)
    return accent2speaker


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 make_dataset_vctk.py [data root directory (VCTK-Corpus)] [h5py path] '
                '[training proportion]')
        exit(0)

    root_dir = sys.argv[1]
    h5py_path = sys.argv[2]
    proportion = float(sys.argv[3])

    accent2speaker = read_speaker_info(os.path.join(root_dir, 'speaker-info.txt'))
    filename_groups = defaultdict(lambda : [])
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, 'wav48/*/*.wav')))
        for filename in filenames:
            # 切分成一個個group
            sub_filename = filename.strip().split('/')[-1]
            # 格式為: p{speaker}_{sid}.wav
            speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            # 挑出所選取的口音
            if speaker_id not in accent2speaker['Mixture']:
                continue
            print('processing {}'.format(speaker_id))
            train_size = int(len(filenames) * proportion)
            for i, filename in enumerate(filenames):
                sub_filename = filename.strip().split('/')[-1]
                # 格式為: p{speaker}_{sid}.wav
                speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
                #以spectrograms形式存於.h5files裡面並同時劃分訓練測試集
                _, lin_spec = get_spectrograms(filename)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}', \
                    data=lin_spec, dtype=np.float32)
