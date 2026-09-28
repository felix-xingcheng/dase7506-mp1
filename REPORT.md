# DASE7506 Project 1 Report

**Small Language Model Challenge**

Student ID: 3036800967  
Name: Si Xingcheng

## 1. Introduction

The task is to train a small language model from random weights on the supplied WikiText-2 text, then lower the test bits per byte (BPB). Lower is better. The course baseline is a 4-block GPT with width 128, four attention heads, and a context of 256. It uses a learned position vector for each of the 256 seats, a GELU feed-forward layer, and 1,200 training steps. On my machine that baseline scored **2.102 test BPB** after **2.072 validation BPB**.

I kept the data, the tokenizer, and `evaluate.py` unchanged. Development decisions used the validation split. I ran the test split only after I had stopped changing the model. The submitted checkpoint is `code/checkpoints/depth8-4k.pt`, and its test score is **1.705 BPB**.

## 2. Method

The code path for my model is `student.py`. It still has to expose `forward` for training and `predict_log_probs` for scoring, with context 256 and a vocabulary of 2048. I changed three things in sequence, and then one combined training run.

### 2.1 Rotary positions instead of a position table

The baseline adds a learned vector for seat 0, another for seat 1, and so on. Those rows are independent, so the model has to memorise 256 positions and is not told that seat 5 is next to seat 6. I removed that table. After the query and key projections, each head's 32 features are split into 16 pairs and rotated by an angle that depends on the seat index. Values are not rotated. The attention score then depends on how far apart two seats are, because the two rotations leave a difference of angles. This is the rotary encoding from Su et al. (2021). It also deletes the position table, so this model has 1,055,488 parameters instead of the baseline's 1,088,256.

### 2.2 A gated feed-forward layer

The baseline feed-forward layer maps 128 features to 512, applies GELU, and maps back to 128. GELU looks at each number on its own. I replaced it with SwiGLU (Shazeer, 2020). The same 128 features go through two linear layers. One output is the content. The other goes through SiLU and is used as a gate. The two vectors of length 341 are multiplied element by element, and a third linear layer maps 341 back to 128 so the residual add still matches. I set the middle width to \(8/3\) of 128, which is 341, so this layer has about the same number of weights as the original 128-512-128 layer. The comparison is then about the gate, not about adding a much larger layer. With the rotary model, the parameter count stays about 1.056 million.

### 2.3 Wider training for the final model

The two changes above were measured at the baseline budget: 1,200 steps, batch size 32, seed 17, so 9,830,400 next-token targets. For the submitted model I also:

- stacked 8 blocks instead of 4 (`configs/depth8.json`), still at width 128;
- trained for 4,000 steps, which is 32,768,000 targets;
- set AdamW's coefficients to `(0.9, 0.95)` instead of the PyTorch default `(0.9, 0.999)`.

The learning-rate shape is the same as the baseline: linear warmup, then a cosine decay from 0.001 down to 0.0001. Because this run is longer than 1,200 steps, the warmup lasts 400 steps rather than 100, and the cosine is stretched over 4,000 steps. I did not increase the width. A wider model would have raised the scoring cost faster than extra layers, and I wanted to stay inside the time limit.

These three training choices were made together. I can say that the package works. I cannot say how much of the last gain came from depth alone, from the extra steps alone, or from the AdamW coefficient.

## 3. Results

All of the 1,200-step runs used seed 17, batch size 32, and the same learning-rate peak. Validation BPB was the number I looked at while changing the model. Test BPB is reported for the baseline and for the frozen final model.

| Run | Depth | Steps | Parameters | Validation BPB | Test BPB |
|---|---:|---:|---:|---:|---:|
| Baseline | 4 | 1,200 | 1,088,256 | 2.072 | 2.102 |
| Rotary positions | 4 | 1,200 | 1,055,488 | 1.923 | — |
| Rotary + SwiGLU | 4 | 1,200 | 1,055,656 | 1.840 | — |
| Submitted model | 8 | 4,000 | 1,848,912 | 1.674 | 1.705 |

The test JSON for the submitted model is `code/results/depth8-4k_test_cpu_fp32.json`. The exact test BPB is 1.7047095379715733. I did not run the test set on the two intermediate models, because those runs were for choosing the method on validation.

## 4. What the comparisons show

From the baseline to the rotary model, the only structural change is the position representation, at the same number of training targets. Validation BPB falls from 2.072 to 1.923. That is the comparison for the position change. My reading is that a relative angle is more useful here than 256 unrelated position rows, and it costs fewer parameters.

From the rotary model to rotary + SwiGLU, depth, width, and the training budget stay the same, and the feed-forward width is matched in parameter count. Validation BPB falls from 1.923 to 1.840. That is the ablation of the gate. The gate can turn a hidden feature down for the current vector even when the content value itself is large. GELU cannot do that, because it only sees one number at a time.

The submitted model then goes from 1.840 to 1.674 on validation, and scores 1.705 on the test set. That step changes depth, the number of steps, and the AdamW coefficient at once. It shows that a deeper model trained for longer still improves on this data. It is not a clean ablation of any one of those three knobs. If I had more time I would repeat the 8-layer model at 1,200 steps, and the 4-layer SwiGLU model at 4,000 steps, so those effects could be separated.

The test score is a bit worse than the validation score (1.705 against 1.674). The baseline has the same pattern (2.102 against 2.072). I treat that as the usual gap between the two splits, not as a sign that I tuned on the test text.

## 5. Cost and the scoring limits

The limits are 5 times the baseline CPU scoring time, 4 GiB peak evaluation memory, and 64 MiB of uncompressed inference files.

On my CPU, the baseline test pass took 8.47 seconds and the submitted model took 16.45 seconds. That is about 1.9 times the baseline, under the 5-times line. The checkpoint is 7,433,220 bytes, about 7.1 MiB, under 64 MiB. The course scorer does not record CPU peak memory (the JSON field stays 0 on CPU). The weight file is 7.1 MiB and the scored batches are small, so the run is not close to 4 GiB.

Training the submitted model on an RTX 3060 Laptop took about 151 seconds in bfloat16, plus about 12 seconds to load the data. The 4-layer baseline training on the same GPU took about 17 seconds. The extra cost is the longer run and the extra four blocks. Ranked scoring was still done in FP32 on CPU.

## 6. Limits of this write-up

I understand the two structural changes well enough to explain them: rotary positions change how queries and keys are paired, and the gate multiplies one linear projection by a SiLU projection. The last jump in score is real, but it is a bundle of changes. I also did not try a wider model, dropout, or a different context. The context and the vocabulary are fixed by the scorer anyway.

The report and the README were drafted with AI assistance, as stated in `README.md`. The training commands, the checkpoints, and the CPU test command were run locally.

## References

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021.

Noam Shazeer. GLU Variants Improve Transformer. arXiv:2002.05202, 2020.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models. arXiv:1609.07843, 2016.
