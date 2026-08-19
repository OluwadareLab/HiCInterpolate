[![Hits](https://hits.sh/github.com/OluwadareLab/HiCInterpolate.svg)](https://hits.sh/github.com/OluwadareLab/HiCInterpolate/)

# HiCInterpolate: 4D Spatiotemporal Interpolation and Analysis of Hi-C Data for Dynamic Genome Architecture.

![HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/hicinterpolate.png)

**HiCInterpolate** is a deep learning–based architecture for **4D spatiotemporal interpolation of Hi-C data**. Given two time-point Hi-C contact matrices, it predicts high-resolution intermediate states while preserving key biological features such as **TADs** and **chromatin loops**.

- **Architecture:** Deep learning flow predictor with a U-Net encoder–decoder.
- **Analysis:** Integrated pipeline for A/B compartments, chromatin loops, and 3D genome structure.
- **Performance:** Validated using PSNR, MS-SSIM, and HiCRep.
- **Biological Performace:** Validated with A/B compartment, Chromatin Loops, TADs, 3D Structures.
- **Explainability:** Explain dynamics of 3D genomic architecture in early mammalian cells development.


### Pretrained Models

Weights are in [`models/`](https://github.com/OluwadareLab/HiCInterpolate/tree/main/models) (Git LFS). Use the checkpoint that matches the resolution you will interpolate.

**Best models by resolution:**

***25KB (default)***
[HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/models/hicinterpolate.pt)
**Latest Snapshot:**
[Snapshot (100 epochs)](https://github.com/OluwadareLab/HiCInterpolate/blob/main/models/hicinterpolate_snapshot.pt)

***10KB***
[HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/models/hicinterpolate_10000.pt)
**Latest Snapshot:**
[Snapshot (100 epochs)](https://github.com/OluwadareLab/HiCInterpolate/blob/main/models/hicinterpolate_snapshot_10000.pt)

***5KB***
[HiCInterpolate](https://github.com/OluwadareLab/HiCInterpolate/blob/main/models/hicinterpolate_5000.pt)
**Latest Snapshot:**
[Snapshot (100 epochs)](https://github.com/OluwadareLab/HiCInterpolate/blob/main/models/hicinterpolate_snapshot_5000.pt)

---

## Documentation
Please see the [wiki](https://github.com/OluwadareLab/HiCInterpolate/wiki) for an extensive documentation.

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
