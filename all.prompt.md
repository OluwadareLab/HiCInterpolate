Position: Senior Research Scientist, Expert: Computer Vision, Data: Genomic Data Format: Hi-C
Stack: Python, Torch, ML, Genomic Data Analysis, Temporal Image Analysis

Always use
  - Communication: caveman ultra
  - Agent Reply: rtk, 
  - Index: codegraph
  - AI model: select optimal model, reduce token usage

Problem: Temporal interpolation of Hi-C data
Description: Utilize, x_0 and x_1 and predict x_0.5 preserving sparsity, biological relevence and interpretability, crisp and temporal consistency.
Evaluation Metrics: PSNR, SSIM, SCC, LPIPS, FID, GenomeDISCO, HiCRep
Best Model: monitor SCC
Report: evaluation metrics, loss, lr. Report in: log, csv, std out

File Structure:
- hicinterpolate.py: main script
- src\
  - train_lib.py: training and evaluation functions
  - data_loader\
    - load_data.py: load Hi-C temporal data
  - interpolation\
    - model.py: interpolation architecture
  - feature_encoder\
    - model.py: encoder
  - flow_predictor\
    - model.py: flow predictor
  - feature_decoder\
    - model.py: decoder
  - loss\
    - models.py: loss functions
  - metric\
    - metrics.py: evaluation metrics
  - misc\
    - plots.py: plotting functions for evaluation metrics
- configs\ : yaml files for training and evaluation configurations

Step 1: read architecture and implementation. current config file: config_dilated_25k_128.yaml.
Step 2: knowledge: U-Net, Flow prediction, skip connections, residual, cost volume, multi-scale features
Step 2: preserve current design and update parts as:
  Input; Feature Extractor; Encoder; Flow Predictor; Decoder; Projection; Output
  x_0,x_1:B,1,128x128; ; ; ; ; ; (16, k=1) -> (1, k=1) -> x_0.5:B,1,128x128
  ; cat((128, k=1), (128, k=3) (128, k=5)); ; ; ; (128, k=1) -> (64, k = 3) -> (32, k=5);
  ; ; (256 k=1) ; (256, k=1) + upsampled residue ; (256, k=1) + upsampled + skip connections from encoder; ;
  ; ; (128, k=3) ; (128, k=3) + upsampled residue ; (128, k=3) + upsampled + skip connections from encoder ; ;
  ; ; (64, k=3) ; (64, k=3) + upsampled residue ; (64, k=3) + upsampled + skip connections from encoder ; ;
  ; ; (32, k=5) ; (32, k=5) + upsampled residue ; (32, k=5) + upsampled + skip connections from encoder ; ;
  ; ; (16, k=7) ; (16, k=7) + upsampled residue ; (16, k=7) + upsampled + skip connections from encoder ; ;

Test pipeline: use conda env: hicinterpolate, dataset: 0.0125, target: end-to-end flow, no error
Rule: i. do not update aggresively, ii. understand first, iii. preserve architectural module, e.g. learnable time factor, iv. update convolutions v. better option? ask first. v. read optimally, reply less, use less and focus on context


