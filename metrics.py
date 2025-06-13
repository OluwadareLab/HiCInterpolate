from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

def calculate_psnr(pred, target):
    return peak_signal_noise_ratio(pred, target, data_range=1.0)

def calculate_ssim(pred, target):
    return structural_similarity_index_measure(pred, target, data_range=1.0)
