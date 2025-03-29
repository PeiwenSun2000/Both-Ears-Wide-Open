# BEWO-1M: Open Source Spatial Audio Dataset

## Introduction

To better facilitate the advancement of multimodal guided spatial audio generation models, we have developed a dual-channel audio dataset named Both Ears Wide Open 1M (BEWO-1M) through rigorous simulations and GPT-assisted caption transformation.

Totally, we constructed 2.8k hours of training audio with more than 1M audio-text pairs and approximately 17 hours of validation data with 6.2k pairs.

## Dataset Overview

BEWO-1M is a large-scale, simulation-based, and GPT-assisted dataset, with abundant soundscapes and descriptions even including moving and multiple sources.

## Data Sources

The dataset is constructed from the following publicly available sources:

1. **[WavCaps](https://huggingface.co/datasets/cvssp/WavCaps)**  
   - A ChatGPT-assisted weakly-labeled audio captioning dataset.  
   - Sources: **FreeSound**, **BBC Sound Effects**, **SoundBible**, and **AudioSet Strongly-labeled Subset**.  

2. **[AudioCaps](https://audiocaps.github.io/)**  
   - A large-scale dataset of audio clips paired with human-written captions.  

3. **[VGGSound](https://github.com/hche11/VGGSound)**  
   - A large-scale audio-visual dataset with audio clips sourced from YouTube videos.  

4. **[ESC-50](https://github.com/karolpiczak/ESC-50)**  
   - A labeled dataset of 2,000 5-second audio recordings across 50 classes.  
   - Categories include animals, natural soundscapes, human sounds, domestic sounds, and urban noises.

## Data Format

The dataset is provided in `JSONL` format, with each line representing one data sample. Below is an explanation of the fields used in the dataset:

### Common Fields

- **`audio_name`**: A unique identifier for each audio sample.  
  - Example: `"M4add9dc5e025a30c39032b4c20a408d3"`.

- **`meta`**: Metadata about the audio source. Provides information about the dataset and file name.  
  - Example: `["ESC50&3-68630-A-40", "vggsound&as8KNZb6Mfs_90000_100000"]`.

- **`caption`**: A textual description of the audio clip, highlighting the scene, events, or sound properties.  
  - Example: `"The helicopter is moving from front right to front left at a moderate speed, while the orchestra is on the right side of the scene."`

- **`room_size`**: A 3D vector representing the room's dimensions (length, width, height).  
  - Example: `[100.185, 105.025, 109.851]`.

- **`micro_pos`**: Position of the microphone array (two microphones) in the scene, represented as `[[x1, x2], [y1, y2], [z1, z2]]` coordinates.
  - Example: `[[51.56, 51.73], [61.66, 61.66], [60.90, 60.90]]`.

- **`start`** and **`end`**: The spatial positions of the audio source at the start and end of the clip. Both are 4D arrays `[x, y, z, ratio]`, where the last value represents the ratio of the audio source's distance to the microphone to the distance to the nearest room boundary. Usually, you can ignore the last value.
  - Example (start): `[[56.44, 72.91, 60.90, 0.28], [88.61, 70.74, 60.90, 0.87]]`.  
  - Example (end): `[[50.31, 70.57, 60.90, 0.20], [39.75, 72.55, 60.90, 0.37]]`.

- **`RT60`**: The reverberation time (RT60) of the room, indicating how quickly sound decays. Unit: seconds.
  - Example: `0.366`.

- **`change_step`**: Time ratios at which the audio source moves instantly during the clip.  
  - Example: `[0, 0.28]`.

- **`raw_start`** and **`raw_end`**: Indices representing the spatial direction of the audio source.  
  - Example: `raw_start: [3, 5]` (where `1` is directly to the left, `3` is directly in front, and `5` is directly to the right).

- **`start_angle`** and **`end_angle`**: Angles of the audio source relative to the microphone at the start and end of the clip.  
  - Example: `start_angle: [66.91, 13.79]`.  (where `0` is directly to the right, `90` is directly in front, and `180` is directly to the left). 

- **`raw_speed`**: Descriptive labels for the speed of the audio source.  
  - Example: `["still", "moderate"]`.

- **`speed_ratio`**: Numeric values representing the relative speed of the audio source. The larger the value, the slower the audio source moves.
  - Example: `[0, 0.488]`.

- **`move_start_time`**: Frame indices indicating when the audio source begins to move.  
  - Example: `[0, 5980]`.

- **`raw_audio_path`**: File paths to the original audio clips.
  - Example: `["/data/path/ESC50/3-68630-A-40.wav", "/data/path/VGGSound/as8KNZb6Mfs.wav"]`.

## Structure
```
BEWO_1M
├── BEWO_SS_Audio_v1
│   ├── audiocaps_single_test
│   ├── audiocaps_single_train
│   ├── audiocaps_single_val
│   └── full_single
├── BEWO_SS_Annotation_v1
│   ├── audiocaps_single_test.jsonl
│   ├── audiocaps_single_train.jsonl
│   ├── audiocaps_single_val.jsonl
│   └── full_single.jsonl
├── BEWO_SD_Annotation_v1
│   ├── audiocaps_move_test.jsonl
│   ├── audiocaps_move_train.jsonl
│   ├── audiocaps_move_val.jsonl
│   └── full_move.jsonl
├── BEWO_DS_Annotation_v1
│   ├── audiocaps_double_test.jsonl
│   ├── audiocaps_double_train.jsonl
│   ├── audiocaps_double_val.jsonl
│   └── full_double.jsonl
├── BEWO_DS_Audio_v1
│   ├── audiocaps_double_test
│   ├── audiocaps_double_train
│   ├── audiocaps_double_val
│   └── full_double
├── BEWO_SD_Audio_v1
│   ├── audiocaps_move_test
│   ├── audiocaps_move_train
│   ├── audiocaps_move_val
│   └── full_move
├── BEWO_Mix_Audio_v1
│   ├── audiocaps_mix_test
│   ├── audiocaps_train_mix
│   ├── audiocaps_val_mix
│   └── full_mix
├── BEWO_Mix_Annotation_v1
│   ├── audiocaps_mix_test.jsonl
│   ├── audiocaps_train_mix.jsonl
│   ├── audiocaps_val_mix.jsonl
│   └── full_mix.jsonl
├── BEWO_RW_Audio_v1
├── BEWO_RW_Annotation_v1
│   └── BEWO_RW_Annotation_v1_meta.csv
├── BEWO_AIP_IA_v1
│   ├── image_test_matched
│   ├── image_train_matched (find in Coco2017 from triplet_*.csv)
│   ├── human_test.csv
│   ├── triplet_0.9.csv (for training option)
│   └── triplet_0.85.csv (for training option)
├── BEWO_BBox_v1 (for evaluation only)
├── BEWO_Click_v1 (for evaluation only)
└── readme.md
```

## Download Link

<!-- 🚀**README**: Please refer to [Github](https://github.com/PeiwenSun2000/Both-Ears-Wide-Open) for usage.

🟠**SS-set (Audio)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) (161GB) or [Baidu Disk](https://pan.baidu.com/s/1eVndQwL5PkjQj9CHEn7s7g?pwd=eccv) (161GB)

🟠**SS-set (Annotation)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🟢**SD-set (Audio)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) (92GB) or [Baidu Disk](https://pan.baidu.com/s/1eVndQwL5PkjQj9CHEn7s7g?pwd=eccv) (92GB)

🟢**SD-set (Annotation)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🔵**DS-set (Audio)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) (113GB) or [Baidu Disk](https://pan.baidu.com/s/1eVndQwL5PkjQj9CHEn7s7g?pwd=eccv) (113GB)

🔵**DS-set (Annotation)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🔴**Mix-set (Audio)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) (150GB) or [Baidu Disk](https://pan.baidu.com/s/1eVndQwL5PkjQj9CHEn7s7g?pwd=eccv) (150GB)

🔴**Mix-set (Annotation)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🟡**RW-set (Audio for Evaluation Only)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) (16.1GB) or [Baidu Disk](https://pan.baidu.com/s/1eVndQwL5PkjQj9CHEn7s7g?pwd=eccv) (16.1GB)

🟡**RW-set (Annotation for Evaluation Only)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🟤**Audio-Image Pairs (Audio and Image)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) (16.1GB) or [Baidu Disk](https://pan.baidu.com/s/1eVndQwL5PkjQj9CHEn7s7g?pwd=eccv) (16.1GB)

🟤**Audio-Image Pairs (Annotation)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🟣**Bounding Box (for Evaluation Only)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv)

🟣**Click (for Evaluation Only)**: Available at [Zenodo](https://zenodo.org/records/12970978/files/RefAVSBench.tar.gz?download=1) or [Baidu Disk](https://pan.baidu.com/s/1UJjtn3KKcpKAMdE6vOPFvQ?pwd=eccv) -->


Huggingface Space: https://huggingface.co/datasets/spw2000/BEWO-1M

Baidu: https://pan.baidu.com/s/1KEnlCGadhd_51vPWjrF5fQ?pwd=temp
Password: temp

Note: The password for Baidu Disk is set to `temp`. Enter the password to initiate the download.

## Usage

Find in data format.

## Licence

We distribute the metadata dataset under the most common [Creative Common CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license, which poses no particular restriction.

## Citation
If you find it helpful, please feel free to cite our paper.
```
@article{sun2024both,
  title={Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation},
  author={Sun, Peiwen and Cheng, Sitong and Li, Xiangtai and Ye, Zhen and Liu, Huadai and Zhang, Honggang and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2410.10676},
  year={2024}
}
```
