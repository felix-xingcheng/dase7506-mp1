# DASE7506 Project 1 Report

**Small Language Model Challenge**

Student ID: 3036800967  
Name: Si Xingcheng

## 1. Introduction

The task is to train a small language model from random weights on the supplied WikiText-2 text, then lower the test bits per byte (BPB). Lower is better. The course baseline is a 4-block GPT with width 128, four attention heads, and a context of 256. It uses a learned position vector for each of the 256 seats, a GELU feed-forward layer, and 1,200 training steps. On my machine that baseline scored **2.102 test BPB** after **2.072 validation BPB**.

I kept the data, the tokenizer, and `evaluate.py` unchanged. Development decisions used the validation split. I ran the test split only after I had stopped changing the submitted model. The submitted checkpoint is `code/checkpoints/depth10-160-12000.pt`, and its test score is **1.579 BPB**.

## 2. Method

The code path for my model is `student.py`. It still exposes `forward` for training and `predict_log_probs` for scoring, with context 256 and a vocabulary of 2048.

### 2.1 Rotary positions instead of a position table

The baseline adds a learned vector for seat 0, another for seat 1, and so on. Those rows are independent, so the model has to memorise 256 positions and is not told that seat 5 is next to seat 6. I removed that table. After the query and key projections, each head's features are split into pairs and rotated by an angle that depends on the seat index. Values are not rotated. The attention score then depends on how far apart two seats are, because the two rotations leave a difference of angles. This is the rotary encoding from Su et al. (2021). At the original width of 128 this change also deletes the position table, so the parameter count falls from 1,088,256 to 1,055,488.

### 2.2 A gated feed-forward layer

The baseline feed-forward layer maps each position from 128 features to 512, applies GELU, and maps back to 128. GELU looks at each number on its own. I replaced it with SwiGLU (Shazeer, 2020). The same vector goes through two linear layers. One output is the content. The other goes through SiLU and is used as a gate. The two vectors are multiplied element by element, and a third linear layer maps back to the model width so the residual add still matches. For the controlled comparison I set the middle width to \(8/3\) of the model width, so this layer has about the same number of weights as the original expand-by-4 layer. The comparison is then about the gate, not about adding a much larger layer.

### 2.3 Dropout, an untied output, and a larger network

After those two changes I increased capacity and added two standard regularisers, because a wider network on this small training set overfit when I removed the regularisation.

- Dropout of 0.1 is applied to the token embeddings, inside attention, and on both residual branches. It is active only while training. Scoring uses the full network.
- The output layer is not tied to the token embedding, and it has a bias. The embedding has to represent "what this token is". The output layer has to score "what the next token is". Those are related but not the same job. The bias can also learn that some tokens are simply more common.
- The submitted model has width 160, 10 blocks, and 4 heads. The feed-forward multiplier stays at \(8/3\). That is 3,749,448 parameters.

### 2.4 Training

The submitted run uses seed 17, batch size 32, and 12,000 steps, which is 98,304,000 next-token targets. The optimiser is AdamW with coefficients `(0.9, 0.95)`, weight decay 0.1, and gradient clipping at 1. The learning rate warms up over the first 1,200 steps to 0.001, then follows a cosine down to 0.0001. Training was in bfloat16 on an RTX 3060 Laptop. The trainer records validation BPB every 500 steps and saves the weights with the lowest validation BPB. In this run that was the last step.

I also tried a few changes that I did not keep. A width-192 model with a feed-forward width of \(4\times\) the model width, trained for 8,000 steps at batch size 64, drove the training loss down to about 1.75 nats per token while the validation loss stayed near 4.3 nats per token. Validation BPB was 2.034, close to the baseline. That is memorisation of the training text. An exponential moving average of the weights did not improve the width-160 model. A depthwise causal convolution of width 4 in every block also made validation worse, so I removed it.

## 3. Results

Validation BPB is the number I used while changing the model. Test BPB is reported for the baseline and for the frozen submitted model.

| Run | Width | Depth | Steps | Parameters | Validation BPB | Test BPB |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 128 | 4 | 1,200 | 1,088,256 | 2.072 | 2.102 |
| Rotary positions | 128 | 4 | 1,200 | 1,055,488 | 1.923 | — |
| Rotary + SwiGLU | 128 | 4 | 1,200 | 1,055,656 | 1.840 | — |
| Width 160, dropout, tied output | 160 | 8 | 6,000 | 2,801,376 | 1.616 | — |
| Same, untied output | 160 | 8 | 6,000 | 3,131,104 | 1.606 | — |
| Submitted shape, 6,000 steps | 160 | 10 | 6,000 | 3,749,448 | 1.592 | — |
| Submitted shape, 10,000 steps | 160 | 10 | 10,000 | 3,749,448 | 1.556 | — |
| Submitted model | 160 | 10 | 12,000 | 3,749,448 | 1.554 | 1.579 |

The test JSON is `code/results/depth10_160_12000_test_cpu_fp32.json`. The exact test BPB is 1.5792066244052076. I did not run the test set on the intermediate models.

## 4. What the comparisons show

From the baseline to the rotary model, the only structural change is the position representation, at the same number of training targets. Validation BPB falls from 2.072 to 1.923. A relative angle is more useful here than 256 unrelated position rows, and it costs fewer parameters.

From the rotary model to rotary + SwiGLU, depth, width, and the training budget stay the same, and the feed-forward width is matched in parameter count. Validation BPB falls from 1.923 to 1.840. The gate can turn a hidden feature down for the current vector even when the content value itself is large. GELU cannot do that, because it only sees one number at a time.

Untying the output, at width 160, depth 8, dropout 0.1, and 6,000 steps, moves validation BPB from 1.616 to 1.606. The gain is small. It is still in the direction I expected: the output layer can specialise, and the bias can capture token frequency.

Increasing depth from 8 to 10, with the same width, dropout, and untied output, reaches 1.592 at 6,000 steps, 1.556 at 10,000 steps, and 1.554 at 12,000 steps. The last extra 2,000 steps barely move validation. The curve was still inching down at step 10,000, which is why I trained to 12,000, but it had nearly stopped. I would not expect a much lower score from still more steps of this same model.

The test score, 1.579, is a bit worse than the validation score, 1.554. The baseline has the same pattern (2.102 against 2.072). I treat that as the usual gap between the two splits. The test text was not used to choose the method.

A separate 1,200-step run of the 10-block model landed near 1.8. That is not a failure of the architecture. On the 6,000-step curve, validation BPB is still about 1.85 at step 1,000 and about 1.76 at step 1,500. Twelve hundred steps is simply too early for this larger model. The early rotary and SwiGLU comparisons stay at 1,200 steps because that is the baseline budget. The submitted model is a later, longer run, and the report should not pretend that 12,000 steps is the same experiment as those 1,200-step ablations.

## 5. Cost and the scoring limits

The limits are 5 times the baseline CPU scoring time, 4 GiB peak evaluation memory, and 64 MiB of uncompressed inference files.

On my CPU, the baseline test pass took 8.47 seconds and the submitted model took 25.47 seconds. That is about 3.0 times the baseline, under the 5-times line. The checkpoint is 15,040,994 bytes, about 14.3 MiB, under 64 MiB. The course scorer does not record CPU peak memory (the JSON field stays 0 on CPU). The weight file is 14.3 MiB and the scored batches are small, so the run is not close to 4 GiB.

Training the submitted model took about 708 seconds in bfloat16, plus about 11 seconds to load the data. The 4-layer baseline training on the same GPU took about 17 seconds. The extra cost is the longer run, the extra blocks, and the wider layers. Ranked scoring was still done in FP32 on CPU.

## 6. Limits of this write-up

The two early structural changes are clean: rotary positions against the learned table, and SwiGLU against GELU, both at the baseline training budget. The later gains mix width, depth, dropout, an untied output, and a much longer training run. I separated some of those with the rows in the table, but I did not cross every pair. In particular, I did not retrain the final 10-block model at exactly 1,200 steps as the official comparison; the short run I did make shows that 1,200 steps under-trains it.

The report and the README were drafted with AI assistance, as stated in `README.md`. The training commands, the checkpoints, and the CPU test command were run locally.

## References

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021.

Noam Shazeer. GLU Variants Improve Transformer. arXiv:2002.05202, 2020.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models. arXiv:1609.07843, 2016.
