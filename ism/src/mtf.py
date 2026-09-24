from math import pi
from config.ismConfig import ismConfig
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hsys = Hdiff * Hwfe * Hdefoc * Hdet * Hsmear * Hmotion # dummy

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO
        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w

        eps=1e-6

        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        [fnAltxx, fnActxx] = np.meshgrid(fAlt, fAct, indexing='ij')
        f2D = np.sqrt(fnAltxx * fnAltxx + fnActxx * fnActxx)

        fc = D/(lambd * focal)

        fn2D = f2D / (1/w)
        fr2D = f2D / (fc)
        fnAct = fAct / (1/w)
        fnAlt = fAlt / (1/w)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        #TODO

        Hdiff = np.zeros_like(fr2D, dtype=float)

        r = np.abs(fr2D)
        mask = r < 1.0
        r_valid = r[mask]

        Hdiff[mask] = (2.0 / np.pi) * (np.arccos(r_valid) - r_valid * np.sqrt(1.0 - r_valid ** 2))

        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        #TODO
        xi_r = np.abs(fr2D)

        x = np.pi * defocus * xi_r * (1.0 - xi_r)

        j1 = (x / 2.0) - (x ** 3 / 16.0) + (x ** 5 / 384.0) - (x ** 7 / 18432.0)

        # For avoiding 0/0 when x = 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            Hdefoc = np.where(x != 0, 2.0 * j1 / x, 1.0)

        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        #TODO
        xi_r = np.abs(fr2D)

        wfe_term = kLF * (wLF / lambd) ** 2 + kHF * (wHF / lambd) ** 2
        Hwfe = np.exp(-xi_r * (1.0 - xi_r) * wfe_term)

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        #TODO
        Hdet = np.abs(np.sinc(fn2D))

        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        #TODO
        Hsmear_1d = np.sinc(ksmear * fnAlt)

        if Hsmear_1d.ndim == 1:
            Hsmear = np.tile(Hsmear_1d[:, np.newaxis], (1, ncolumns))
        elif Hsmear_1d.shape[1] == 1:
            Hsmear = np.tile(Hsmear_1d, (1, ncolumns))
        else:
            Hsmear = Hsmear_1d

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        #TODO
        Hmotion = np.sinc(kmotion * fn2D)

        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        #TODO
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        mid_alt = nlines // 2
        mid_act = ncolumns // 2

        freq_act = fnAct[mid_alt, :] if fnAct.ndim == 2 else fnAct
        freq_alt = fnAlt[:, mid_act] if fnAlt.ndim == 2 else fnAlt

        # --- 1. Slice ACT ---
        mask_act = (freq_act >= 0) & (freq_act <= 0.50001)
        sort_idx_act = np.argsort(freq_act[mask_act])
        x_act = freq_act[mask_act][sort_idx_act]

        plt.figure(figsize=(10, 5.8))
        plt.plot(x_act, (Hdiff[mid_alt, :][mask_act])[sort_idx_act], label="Diffraction MTF", linewidth=1.2)
        plt.plot(x_act, (Hdefoc[mid_alt, :][mask_act])[sort_idx_act], label="Defocus MTF", linewidth=1.2)
        plt.plot(x_act, (Hwfe[mid_alt, :][mask_act])[sort_idx_act], label="WFE Aberrations MTF", linewidth=1.2)
        plt.plot(x_act, (Hdet[mid_alt, :][mask_act])[sort_idx_act], label="Detector MTF", linewidth=1.2)
        plt.plot(x_act, (Hsmear[mid_alt, :][mask_act])[sort_idx_act], label="Smearing MTF", linewidth=1.2)
        plt.plot(x_act, (Hmotion[mid_alt, :][mask_act])[sort_idx_act], label="Motion blur MTF", linewidth=1.2)
        plt.plot(x_act, (Hsys[mid_alt, :][mask_act])[sort_idx_act], label="System MTF", color="black", linewidth=2.5)

        plt.axvline(x=0.5, color="black", linestyle="--", linewidth=2.0, label="f Nyquist")

        plt.title("System MTF - slice ACT", fontsize=13)
        plt.xlabel("Spatial frequencies f/(1/w) [-]", fontsize=11)
        plt.ylabel("MTF", fontsize=11)
        plt.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)
        plt.xlim(-0.02, 0.52)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="lower left", fontsize=8, framealpha=0.9)
        plt.tight_layout()

        plt.savefig(output_dir / f"system_mtf_act_{band}.png", dpi=300, bbox_inches="tight")
        plt.show()

        # --- 2. Slice ALT ---
        mask_alt = (freq_alt >= 0) & (freq_alt <= 0.50001)
        sort_idx_alt = np.argsort(freq_alt[mask_alt])
        x_alt = freq_alt[mask_alt][sort_idx_alt]

        plt.figure(figsize=(10, 5.8))
        plt.plot(x_alt, (Hdiff[:, mid_act][mask_alt])[sort_idx_alt], label="Diffraction MTF", linewidth=1.2)
        plt.plot(x_alt, (Hdefoc[:, mid_act][mask_alt])[sort_idx_alt], label="Defocus MTF", linewidth=1.2)
        plt.plot(x_alt, (Hwfe[:, mid_act][mask_alt])[sort_idx_alt], label="WFE Aberrations MTF", linewidth=1.2)
        plt.plot(x_alt, (Hdet[:, mid_act][mask_alt])[sort_idx_alt], label="Detector MTF", linewidth=1.2)
        plt.plot(x_alt, (Hsmear[:, mid_act][mask_alt])[sort_idx_alt], label="Smearing MTF", linewidth=1.2)
        plt.plot(x_alt, (Hmotion[:, mid_act][mask_alt])[sort_idx_alt], label="Motion blur MTF", linewidth=1.2)
        plt.plot(x_alt, (Hsys[:, mid_act][mask_alt])[sort_idx_alt], label="System MTF", color="black", linewidth=2.5)

        plt.axvline(x=0.5, color="black", linestyle="--", linewidth=2.0, label="f Nyquist")

        plt.title("System MTF - slice ALT", fontsize=13)
        plt.xlabel("Spatial frequencies f/(1/w) [-]", fontsize=11)
        plt.ylabel("MTF", fontsize=11)
        plt.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)
        plt.xlim(-0.02, 0.52)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="lower left", fontsize=8, framealpha=0.9)
        plt.tight_layout()

        plt.savefig(output_dir / f"system_mtf_alt_{band}.png", dpi=300, bbox_inches="tight")
        plt.show()


