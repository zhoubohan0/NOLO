# NOLO: Navigate Only Look Once 
[![Arxiv](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=google-docs&color=cc0000&logoColor=white)](https://arxiv.org/pdf/2408.01384)
[![Website](https://img.shields.io/badge/Website-blue?style=for-the-badge&logo=react&color=007BFF&logoColor=white)](https://sites.google.com/view/nol0)
[![BibTex](https://img.shields.io/badge/Bibtex-green?style=for-the-badge&logo=firebase&color=4F805D&logoColor=white)](https://scholar.googleusercontent.com/scholar.bib?q=info:VO1B5Z6HOVgJ:scholar.google.com)

## 0. Overview 🙌
We aim to tackle the video navigation problem, whose goal is to train an in-context policy to find objects included in the context video in a new scene.
After watching an 30-second egocentric video, an agent is expected to reason how to reach the target objetct specified by a goal image. **Please refer to our ![website](https://sites.google.com/view/nol0) for videos of real-world deployment**. 


## 1. Install 🚀
### 1.1 Install required packages 🛰️
```bash
conda create -n nolo python=3.9
conda activate nolo
cd nolo
pip install -r docs/requirements.txt
```

### 1.2 Install RoboTHOR and Habitat 🍔
Refer to [RoboTHOR](https://ai2thor.allenai.org/robothor/documentation) to install RoboTHOR and 
[Habitat-sim](https://github.com/facebookresearch/habitat-sim) to install Habitat simulator.


### 1.3 Download Pretrained Models 📑
SuperGlue: https://github.com/magicleap/SuperGluePretrainedNetwork. Place the downloaded checkpoints in `scripts/superglue/weights`. 
GMFlow: https://github.com/haofeixu/gmflow. Place the downloaded checkpoints in `scripts/gmflow/gmflow-pretrained`. 
Detic: https://github.com/facebookresearch/Detic. Place the whole repository in `scripts/Detic`.

## Dataset 📚
### 1. Download Scene Dataset in Habitat 🗂️
Refer to [Habitat-lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) to install Matterport3D datasets. Change the path in `scripts/collect_habitat_all.py` to where the dataset stores.

### 2. Create Dataset 📥
Domain can be chosen from `'robothor' or 'habitat'`.
```bash
python -m scripts.collect_$domain$_all
```
The generated offline datasets will be in the following structure:

```
offline-dataset
├── robothor-dataset
│   ├── 900
│   │   ├── train
│   │   │   ├── FloorPlan1_1
│   │   │   ├── FloorPlan1_2
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── FloorPlan1_5
│   │   │   ├── FloorPlan2_5
│   │   │   ├── ...
├── mp3d-dataset
│   ├── 900
│   │   ├── train
│   │   │   ├── 1LXtFkjw3qL
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── 2azQ1b91cZZ
│   │   │   ├── ...
```

## 3. Offline Reinforcement Learning 🎮
Train a VN-Bert policy using BCQ in `'robothor' or 'habitat'`.
```bash
python -m recbert_policy.train_vnbert --exp_name bcq_rank_0.5_9_SA --domain $domain$
```

## 4. Run Evaluation! 🏆
- Evaluate Random policy in `'robothor' or 'habitat'`:
```bash
bash bash/eval_$domain$_random.sh
```
- Evaluate baseline LMM policy `'gpt4o' or 'videollava'` in `'robothor' or 'habitat'`.
```bash
bash bash/eval_$domain$_baseline.sh  $baseline$
```

​		Notice to provide a API-KEY if use gpt4o for evaluation.

- Evaluate VN-Bert policy (NOLO) in `'robothor' or 'habitat'`. Ablation variants and cross-domain evaluation are also supported.
```bash
bash bash/eval_habitat_policymode.sh  "nolo-bert" $checkpoint_path$ "Q" "SA"
```
## 5. Real World Experiments! 🤖
- Collect random RGB and action sequence
```bash
python scripts/collect_maze.py
``` 
- Decode actions from recorded video:
```bash
python scripts/label_maze.py
```
- Train a policy using BCQ in real world maze environment.
```bash
python -m recbert_policy.train_vnbert_real --exp_name maze
```
- Evaluate the trained policy
```bash
python -m scripts.inference_maze_transformer
```

