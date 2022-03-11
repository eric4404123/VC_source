# 以生成式深度學習達成多語者跨語言之語音轉換
# Multi Target, And Cross-Language Voice Conversion with Generative Deep Learning


本程式碼修改於[Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations](https://arxiv.org/abs/1804.02812).

# 建置環境
- python 3.6+
- pytorch 1.3.0
- h5py 3.1.0
- tensorboardX

# Preprocess 階段
python3 make_dataset_vctk.py data_root_dir h5py_path train_proportion

python3 make_single_samples.py the_h5py_path index_path n_samples seg_len speaker_used_path

- **data_root_dir**: [VCTK資料集路徑](https://datashare.ed.ac.uk/handle/10283/3443)
- **h5py_path**: 路徑用於放置採樣出的feature位置。首先，我們要先新增一個空白文件並取名然後把這一個路徑指向這個空白文件
- **traini_proportion**: 訓練資料佔比，預設0.9
- **the_h5py_path**: 前述的h5py路徑，這裡要注意的是檔案的副檔名要變成.h5
- **index_path**: 儲存所選擇語者個各個語音片段
- **n_samples**: 採樣樣本數，預設500000
- **seg_len**: 所採樣樣本的片段長，預設128
- **speaker_used_path**: 本次訓練所需的語者

# Training 階段
python3 main.py
- **--load_model**: 是否繼續從先前checkpoint繼續訓練
- **-flag**: 在tenosrboard中表示目前階段，預設:train
- **-hps_path**: 超參數集，預設:mixture_10_.json
- **-dataset_path**:採樣出的feature位置(.h5)
- **-output_model_path**: 訓練出的模型儲存位置