# DASE7506 Project 1 — Small Language Model

Student ID: 3036800967  
Name: Si Xingcheng

This repository trains a small GPT on the supplied WikiText-2 split and evaluates it with the course scorer. The submitted model is a 10-block student implementation of width 160. Its full-test score is **1.579 bits per byte** (FP32, CPU).

The file to score, without training again, is:

`code/checkpoints/depth10-160-12000.pt`

## 1. Install

Use Python 3.12. From this repository:

```powershell
cd code
conda create -n mp1 python=3.12 -y
conda activate mp1
```

Install PyTorch 2.7.1 for one device, then the other two packages. Do not install `torch` a second time from `requirements.txt`, because that file also pins `torch` and can replace the build you just chose.

CPU:

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

NVIDIA GPU (this is what I used for training: an RTX 3060 Laptop, CUDA 12.6):

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

Check the model contract:

```powershell
python -m unittest discover -s tests -v
```

No API key and no extra dataset download are required. The text and the tokenizer are already in `code/data/`.

## 2. Reproduce the submitted score

From `code/`, with the checkpoint that is already in the repository:

```powershell
python evaluate.py --checkpoint checkpoints/depth10-160-12000.pt --device cpu --precision fp32 --split test
```

The number to compare is `bpb` in the printed JSON. On my machine this run reported:

- `bpb`: 1.5792066244052076
- `seconds`: 25.469
- `precision`: fp32
- `split`: test

The saved output is `code/results/depth10_160_12000_test_cpu_fp32.json`.

## 3. Train the same model again

Training is only needed if you want to rebuild the checkpoint. The submitted score does not depend on retraining. From `code/`:

```powershell
python train.py --implementation student --config configs/depth10-160.json --device cuda --seed 17 --steps 12000 --batch-size 32 --eval-every 500 --run-dir runs/depth10-160-12000
```

Use `--device cpu` if you have no GPU. The run writes `runs/depth10-160-12000/checkpoint.pt`. Training on my RTX 3060 Laptop took about 708 seconds in bfloat16. The ranked evaluation must still be FP32 on CPU, using the command in Section 2.

The trainer keeps the weights with the lowest validation BPB. In this run that happened to be the final step, 12000.

The baseline, for comparison, is the untouched course model:

```powershell
python train.py --implementation model --config configs/baseline.json --device cuda --seed 17 --steps 1200 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

My baseline test score was 2.102 BPB. Logs for the baseline and for the intermediate models are in `code/results/`.

## 4. What the student model is

`student.py` keeps the course interfaces (`forward` and `predict_log_probs`, context 256, vocabulary 2048). Relative to `model.py` it:

- removes the learned position table and applies rotary position encoding to queries and keys;
- replaces the GELU feed-forward layer with a SwiGLU layer whose hidden width is `8/3` of the model width;
- drops 10% of activations during training (embedding, attention, and both residual branches);
- does not tie the output layer to the token embedding, and gives the output layer a bias;
- reads depth, width, dropout, and the feed-forward multiplier from the config. The submitted run uses `configs/depth10-160.json` (10 blocks, width 160, 4 heads).

`train.py` uses AdamW with `betas=(0.9, 0.95)`. For a run longer than 1,200 steps, the learning-rate warmup is one tenth of the step count (1,200 steps when training for 12,000). The cosine decay still ends at one tenth of the peak learning rate, 0.0001.

## 5. AI assistance and reused work

I used Cursor's AI assistant while learning the baseline, while writing and checking the rotary, SwiGLU, dropout, and untied-output code, while choosing the depth and training length, and while drafting this README and `REPORT.md`. I ran the training and the CPU evaluation myself. The implementation and the reported numbers are my responsibility.

The following are not my own inventions:

- the baseline GPT, trainer, evaluator, tokenizer, and WikiText-2 split, supplied by the course;
- rotary position embeddings (Su et al., 2021);
- SwiGLU (Shazeer, 2020);
- dropout and an untied output layer, which are standard transformer choices;
- AdamW with a second-moment coefficient of 0.95, which is a common setting in language-model training.

A short report of the method and the comparisons is in `REPORT.md`.

## 6. Data

WikiText-2 comes from Merity, Xiong, Bradbury, and Socher, "Pointer Sentinel Mixture Models" (https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The upstream dataset is released under CC BY-SA 3.0 and the GNU Free Documentation License. Those notices stay with the copies in `code/data/`.
