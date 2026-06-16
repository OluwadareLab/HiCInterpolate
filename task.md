- Use
    - caveman ultra
    - rtk
    - codegraph
- Permission
    Allow this session: Yes
- Position
    - Senior Research Scientist expert in machine learning, computer vision, temporal image generation, sparse genomic data e.g. Hi-C, single-cell Hi-C analysis.
- Stack: 
    - Python, PyTorch, ML, Genomic Data Analysis, Temporal Image Analysis
- Problem statement
    - temporal image interpolation
    - maintain structure and sparsity
    - predict crips, sharpe and texture image
- Background
    - Genomic Hi-C contact matrix.
    - Hi-C contact matrix generally sparse
    - Sparsity appears noise but they are real interactions
    - Predict intermediate Hi-C contact matric, say x_1 (y_0.5) from two input Hi-C contact matrix, say x_0, x_2
- Target
    - increase ssim, scc, HiCRep score 
    - expected score > 0.95
    - prevent oversmooting
    - keep ground truth like sparsity, crisp, texture and sharpness
- Architecture
    - Main caller: hicinterpolate.py
    - Training setup: src/train_lib.py
    - Configuration: configs/config_dilated_25k_128.yaml; Note: many parameters are unused
    - Feature Extractor: src/interpolator/model.py; classes: FeatureBlock, FeatureExtraction
    - U-Net Encoder: src/feature_encoder/model.py
    - Flow Predictor: src/flow_predictor/model.py
    - U-Net Decoder: src/feature_decoder/model.py
    - Feature Projection: src/interpolator/model.py; classes: FeatureProjection
    - Output Projection: src/interpolator/model.py; classes: OutputProjection with logits and mask
    - Learning rate: 2e-4 to 1e-5
    - Optimizer: src/train_lib.py AdamW LINE: 61-69
    - Scheduler: src/train_lib.py LambdaLR/CosineAnnealingLR with warmup LINE:70-91
    - Loss Function: src/loss/model.py Combined Loss defined in configuration file
- Test run (if required)
    - use conda environment: hicinterpolate
    - check syntex
    - check flow with very minimal data
    - do not test unnecessary
- Rule
    - think deeply
    - do not look on unnecessary files
    - think first, provide outcome, ask for implementation if required (at leaset one time), update codebase
    - save updated architectur in a md file: look.md
    - do not show what you thinking or doing, think or do yourself internally if required
    - do not update codebase

Issues
    - Decoder loop does not chain outputs —  FeatureDecoder.forward()  re-reads from  xs  each iteration instead of threading the previous output; only the single finest-level block effectively runs; coarser decoder blocks are dead.
    - Symmetry not enforced at inference — Hi-C matrices are symmetric by definition;  SymmetryLoss  weight is only 0.1 during training and no hard  0.5*(pred + pred.T)  is applied at output.
    - OutputProjection  logit re-derivation is unsound —  pred  and  pred_mask  are each already  sigmoid(...)  outputs; multiplying them and re-deriving logits via  log(p/(1-p)) creates a heuristic that is not a proper probability, producing noisy and disconnected gradients for both heads.
    - Final encoder bottleneck discarded —  EncoderBlock  returns  (latent, skip) ; the encoder loop collects only  skip s and discards the final  latent  (the most spatially compressed representation at  H/16 ), losing the deepest global context.
    - Cost volume uses  reflect  padding —  FlowEstimationBlock._cost_volume  pads  x2  with  mode="reflect" ; for Hi-C, the diagonal and boundary regions have specific sparsity structure; reflect-padding invents artificial contacts that bias cost volume computation.
    - No attention for long-range TAD structure — all convolution kernels are at most 5×5; TAD boundaries are full-matrix diagonal features that pure local convolution cannot capture, which directly limits HiCRep/SCC ceiling.
    - VGG perceptual / style loss domain mismatch — VGG19 is trained on natural images; Hi-C matrices are log-normalized sparse genomic matrices; the resulting gradients carry ImageNet texture semantics, not genomic contact semantics.
    - max_disp=4  is fixed across all pyramid levels — at the coarsest level the search window covers a large relative fraction (appropriate); at the finest level (128×128 input) it covers only ~3% of the image, which may be too small for TAD-level displacements.
    - No input normalization contract — model assumes input range  [0,1]  (sigmoid output, SSIM  data_range=1.0 , focal clamp); no internal assertion or normalization layer enforces this; a mismatch silently degrades all losses.
    - Loss curriculum not staged — all losses run from epoch 0;  bce / focal  on the mask head require the head to stabilize first before they provide useful signal; premature mask supervision may destabilize early training.

Task
    - fix the issues carefully
    - think about adding dice loss if there is a probability to improve the resuts


