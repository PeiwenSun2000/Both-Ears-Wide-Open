
# Audio ITD Analysis Automation Tool

This tool automates the process of analyzing Interaural Time Difference (ITD) between two audio folders.

## Prerequisites

Follow the installation of [StereoCRW](https://github.com/IFICL/stereocrw) and [AudioLDM_eval](https://github.com/haoheliu/audioldm_eval) in the same enveronment.

(or refer to the [environment.yml](environment.yml) file.)


## Usage

The two folders contain the same number of audio files with identical names.

Run the automation script with two folder paths as arguments:

```bash
python automate_process.py --folder1 /path/to/first/folder --folder2 /path/to/second/folder
```

The script will:
1. Generate CSV files for both folders
2. Run visualization for each folder
3. Calculate ITD distances between the folders

## Example

```bash
python automate_process.py --folder1 ./data/folder1 --folder2 ./data/folder2

Output:

Results for CRW mode:
KL-divergence 1.1118132758501538
[info] Overall ITD MSE: 37.614105127440794

Results for GCC mode:
KL-divergence 1.197748563271088
[info] Overall ITD MSE: 39.38635211678856

[info] fsad_score: 0.22983144932751726

```

## Important Notes

- Input folders should contain audio files
- The script will create temporary CSV files (`temp_1.csv` and `temp_2.csv`)
- Pkl files will be saved in the `temp_1` and `temp_2` directory


## Output

The script will generate:
- Distance calculations between the two folders
- PKL files containing ITD analysis data

## License

MIT

# Reference

If you find this repo useful, please cite our papers:

```
@article{sun2024both,
  title={Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation},
  author={Sun, Peiwen and Cheng, Sitong and Li, Xiangtai and Ye, Zhen and Liu, Huadai and Zhang, Honggang and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2410.10676},
  year={2024}
}
```

Please also cite the previous paper if you use the code in this repo. Thanks again for their great work.
```
@inproceedings{chen2022sound,
  title={Sound localization by self-supervised time delay estimation},
  author={Chen, Ziyang and Fouhey, David F and Owens, Andrew},
  booktitle={European Conference on Computer Vision},
  pages={489--508},
  year={2022},
  organization={Springer}
}
```
