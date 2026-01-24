# HiCInterpolate: 4D Spatiotemporal Interpolation of Hi-C Data for Genome Architecture Analysis.

![HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/resources/figure1.png)

In this study, we developed HiCInterpolate, a 4D spatiotemporal interpolation architecture that accepts two timestamp Hi-C contact matrices to interpolate intermediate Hi-C contact matrices at high resolution. HiCInterpolate predicts the intermediate Hi-C contact map using a deep learning-based flow predictor, and a feature encoder and decoder architecture similar to U-Net. In addition, HiCInterpolate supports downstream analysis of multiple 3D genomic features, including A/B compartments, chromatin loops, TADs, and 3D genome structure, through an integrated analysis pipeline. Across multiple evaluation metrics, including PSNR, SSIM, GenomeDISCO, HiCRep, and LPIPS, HiCInterpolate achieved consistently strong performance. Biological validation further demonstrated preservation of key chromatin organization features, such as chromatin loops, A/B compartments, and TADs. Together, these results indicate that HiCInterpolate provides a robust computer vision–based framework for high-resolution interpolation of intermediate Hi-C contact matrices and facilitates biologically meaningful downstream analyses.

---

## Documentation
Please see the [wiki](https://github.com/OluwadareLab/HiCInterpolate/wiki/HiCInterpolate) for an extensive documentation.

---

### Developers:

H M A Mohit Chowdhury<br>
Department of Computer Science and Engineering<br>
University of North Texas<br>
Email: h.m.a.mohitchowdhury@my.unt.edu<br>
<br>

### Contact:

Dr. Oluwatosin Oluwadare <br>
Department of Computer Science and Engineering<br>
University of North Texas<br>
Email: Oluwatosin.Oluwadare@unt.edu <br>

***
### [OluwadareLab, University of North Texas](https://oluwadarelab.com/)

<div style="background-color: black; padding: 20px; text-align: center; border-radius: 8px;">
  <img src="https://webassets.unt.edu/assets/branding/unt-mobile-logo.svg" 
       alt="UNT" 
       style="max-width: 80%; height: auto; margin-top: 10px;">
</div>
