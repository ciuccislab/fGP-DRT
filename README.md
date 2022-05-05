# Project

## fGP-DRT: finite Gaussian Process Distribution of Relaxation Times

This repository contains some of the source code used for the paper titled *The probabilistic deconvolution of the distribution of relaxation times with finite Gaussian processes*. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2022.140119. The article is available online at [Link](https://doi.org/10.1016/j.electacta.2022.140119) and in the docs folder.

# Introduction
Electrochemical impedance spectroscopy (EIS) is a tool widely used to study the properties of electrochemical systems. The distribution of relaxation times (DRT) is a widely used approach, in electrochemistry, biology and material science, for the analysis of electrochemical impedance 
spectroscopy (EIS) data [1]. Deconvolving the DRT from EIS data is quite challenging because an ill-posed problem needs to be solved [2-5]. Several approaches such as ridge regression, ridge and lasso regression, Bayessian and hierarchical Bayesian, Hilbert transform and Gaussian process methods have been used [2-7]. Gaussian processes can be used to regress EIS data, quantify uncertainty, and deconvolve the DRT. However, previously developed DRT models based on Gaussian processes do not constrain the DRT to be non-negative and can only use the imaginary part of EIS spectra [8,9]. Therefore, we overcome both issues by using a finite Gaussian process approximation to develop a new framework called the finite Gaussian process distribution of relaxation times (fGP-DRT) [10]. The analysis on artificial EIS data shows that the fGP-DRT method consistently recovers exact DRT from noise-corrupted EIS spectra while accurately regressing experimental data. Furthermore, the fGP-DRT framework is used as a machine learning tool to provide probabilistic estimates of the impedance at unmeasured frequencies. The method is further validated against experimental data from fuel cells and batteries. In short, this work develops a novel probabilistic approach for the analysis of EIS data based on Gaussian process, opening a new stream of research for the deconvolution of DRT. 

![Screenshot 2022-02-12 165048](https://user-images.githubusercontent.com/99115272/153704506-9184e95d-4a07-4233-ac7f-cbb4bbdee680.gif)

# Dependencies
numpy

scipy

matplotlib

pandas

# Tutorials
1. **example1_single ZARC Model.ipynb**: this notebook gives detail procedure of how to recover the DRT from the impedance generated using a single ZARC model consisting of a resistance placed in parallel to a constant phase element (CPE) The frequency range is from 1E-4 Hz to 1E4 Hz with 10 points per decade (ppd).
2. **example2_double ZARC Model.ipynb** : this notebook demonstrates how the fGP-DRT can capture overlapping timescales with two ZARC models arranged in series. The frequency range is from 1E-4 Hz to 1E4 Hz with 10 ppd.
3. **example3_single_ZARC_plus_an_inductor.pynb** : this notebook adds an inductor to the model used in "**example1_single ZARC Model.ipynb**"
4. **example4_BLF_pO2_60percent_Temp_500_C.ipynb** : this notebook displays the DRT analysis of the BLF impedance spectra from fuel cell. The real experimental EIS data is read from a csv file, the DRT is predicted by the fGP-DRT model, the complete impedance is, therefore, recovered and compared with the equivalent circuit model (ECM) consisting of two ZARCs
5. **example5_SCFN_3percent_H2O_Temp_500_C.ipynb** : this notebook shows the DRT analysis of the SCFN impedance spectra from real experiment. Also the real EIS data is read from a csv file, the DRT is predicted by the fGP-DRT model, the real and imaginary components of the impedance are recovered and compared with the 2ZARCs ECM. 

# Citation

```
@article{maradesa2022probabilistic,
  title={The Probabilistic Deconvolution of the Distribution of Relaxation Times with Finite Gaussian Processes},
  author={Maradesa, Adeleke and Py, Baptiste and Quattrocchi, Emanuele and Ciucci, Francesco},
  journal={Electrochimica Acta},
  pages={140119},
  year={2022},
  publisher={Elsevier}
}

```

# References
[1] Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry.132-139. https://doi.org/10.1016/j.coelec.2018.12.003. 

[2] Wan, T. H., Saccoccio, M., Chen, C., & Ciucci, F. (2015). Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools. Electrochimica Acta, 184, 483-499. https://doi.org/10.1016/j.electacta.2015.09.097.

[3] Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods-a theoretical and experimental study. Electrochimica Acta, 147, 470-482. https://doi.org/10.1016/j.electacta.2014.09.058.

[4] Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439-454. https://doi.org/10.1016/j.electacta.2015.03.123.

[5] Effat, M. B., & Ciucci, F. (2017). Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data. Electrochimica Acta, 247, 1117-1129. https://doi.org/10.1016/j.electacta.2017.07.050.

[6]   Liu, J., & Ciucci, F. (2020). The deep-prior distribution of relaxation times. Journal of The Electrochemical Society, 167 (2) 026-506. https://doi.org/10.1149/1945-7111/ab631a

[7] Liu, J., Wan, T. H., & Ciucci, F.(2020). A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357, 136-864. https://doi.org/10.1016/j.electacta.2020.136864.

[8] Liu, J., & Ciucci, F. (2020). The Gaussian process distribution of relaxation times: A machine learning tool for the analysis and prediction of electrochemical impedance spectroscopy data. Electrochimica Acta, 135316. https://doi.org/10.1016/j.electacta.2019.135316.

[9] Quattrocchi, E., Wan, T. H., Belotti, A., Kim, D., Pepe, S., Kalinin, S. V., Ahmadi, M., and Ciucci, F. (2021). The deep-DRT: A deep neural network approach to deconvolve the distribution of relaxation times from multidimensional electrochemical impedance spectroscopy data. Electrochimica Acta, 139010. https://doi.org/10.1016/j.electacta.2021.139010

[10] Maradesa, A., Py, B., Quattrocchi, E., & Ciucci, F. (2022). The probabilistic deconvolution of the distribution of relaxation times with finite Gaussian processes. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2022.140119.
