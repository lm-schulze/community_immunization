# PoLDaE project: Immunization on networks with community structure
Final project for the course *Physics of Life, Data and Epidemiology* @UNIPD, AY 2025/26. 

In this project, we explore immunization strategies for networks with community structure, focusing especially on algorithms using only local information. We mainly compare [ACQ (Acquaintance immunization)](https://doi.org/10.1103/PhysRevLett.91.247901), [CBF (Community Bridge Finder)](https://doi.org/10.1371/journal.pcbi.1000736), [BHD (Bridge-Hub Detector)](https://doi.org/10.1371/journal.pone.0083489), [BNI-LI (Bridge-Hub Node Identification based on Local Information)](https://doi.org/10.1007/s13278-025-01549-1), and BNI-LI with additional local probing. We analyse them  by performing SIR simulations on both empirical and simulated networks, including a variety of coverages, network modularities, recovery and transmission rates. Part of the simulations are carried out using the [Epidemics on Networks (EoN) python module](https://doi.org/10.21105/joss.01731) (see [here](https://epidemicsonnetworks.readthedocs.io/en/latest/y) for the documentation).

Authors: 
- Jeyran Jamali ([@JeyranJamali](https://github.com/JeyranJamali)): Algorithm comparison on **Empirical networks**
- Laura Schulze ([@lm-schulze](https://github.com/lm-schulze)): Algorithm comparison on **Simulated networks**

References:
- Cohen, R., Havlin, S., & ben-Avraham, D. (2003). Efficient Immunization Strategies for Computer Networks and Populations. Phys. Rev. Lett., 91(24), 247901. https://doi.org/10.1103/PhysRevLett.91.247901
- Salathé M, Jones JH (2010) Dynamics and Control of Diseases in Networks with Community Structure. PLOS Computational Biology 6(4): e1000736. https://doi.org/10.1371/journal.pcbi.1000736
- Gong K, Tang M, Hui PM, Zhang HF, Younghae D, Lai Y-C (2013) An Efficient Immunization Strategy for Community Networks. PLoS ONE 8(12): e83489. https://doi.org/10.1371/journal.pone.0083489
- Zhang, D., Meng, X. & Sheng, J. (2026) An immunization strategy for community networks based on local structural information. Soc. Netw. Anal. Min. 16, 12 . https://doi.org/10.1007/s13278-025-01549-1
- Miller et al., (2019). EoN (Epidemics on Networks): a fast, flexible Python package for simulation, analytic approximation, and analysis of epidemics on networks. Journal of Open Source Software, 4(44), 1731, https://doi.org/10.21105/joss.01731

