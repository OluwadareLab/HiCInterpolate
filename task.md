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
    - Model training is so slow, e.g. for 11652 sample, it takes 10 minutes 
    - Model performance is very very poor, e.g. ssim only 0.26 at the end

Task
    - Read architecture.
    - Do not focus on unused codeblocks or which are not used
    - find root causes for training time delay
    - find root causes for not improving scores
    - As my data is sparse and i want to maintain them reather than washout or oversmoothing, suggest me architectural change only. no code update. give a plan.


