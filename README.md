# About this space
- This is the place where I save my Python projects. 

# Project Index
- Research Paper Model Recreation & Modifying (Exect working hyperparameter pool are not povided)
    - Online Portfolio Selection/Adaptive_Mean_Reversion_Trading_Algo_(Tsang_et_al).py
        - A mean reversion plus robust optimization model that consider transaction cost.
        - Slightly modified the parameter selection logic with EWMA and optimization logic to allow risk seeking behaviour in some regimes.
        - Credit: Tsang et al (2025)
    - Online Portfolio Selection/Dynamic_Mode_Decomposition_Algo_(Li_et_al).py
        - A momenum chasing strategy to increase the weight of high performer exponentially.
        - Exect replicate
        - Credit: Li et al (2026)
    - Online Portfolio Selection/Robust_Momentum_Optimizing_Trading_Algo_(Hybrid).py
        - Combined DMD and robust optimization
        - Credit: Li et al (2026), Tsang et al (2025)
    - Online Portfolio Selection/Adaptive_Dynamic_Mode_Decomposition_(Hybrid).py
        - Combined DMD, robust optimization and dynamic parameter selection with EWMA logic.
        - So far 
        - Credit: Li et al (2026), Tsang et al (2025)
    - Referance list:
        - Li, Jiahao & Zhang, Yong & Zheng, Xiaoteng, 2026. "Dynamic mode decomposition for online portfolio selection task," European Journal of Operational Research, Elsevier, vol. 328(1), pages 349-365.
        - Tsang, M. Y., Sit, T., & Wong, H. Y. (2025). Adaptive robust online portfolio selection. European Journal of Operational Research, 321(1), 214–230.

- Skills Learning Projects (From Latest to Oldest):
    - Time series analysis toolkit
        - A Dynamic FFT Implementation on TSA
            - Build a model that dynamically apply FFT in different window at a list of specified period to better decompose the time series cycles.
            - Include Gaussian fade-in and out option to simulate reallife financal time series pereformance. 
        - Economic cycle indicator
            - Build indicator based on 20y historical data using fast fourier transform. 
    - Economic cycle analysis
        - Apply time series analysis to analysis past economic cycles (S&P 500 is selected as analyzing target). 
        - Identify and label "turning point" dates that have significant contribution of creating each economic cycles.
        - Use LLM to identify key events in these dates and learn the key driver of economic cycles.
    - Time series analysis toolkit
        - Single equity analysis
            - Use SVJ model alongside GARCH(1,1) to visualize upside and downside risk of single stock including volatile microcaps.
        - Portfolio risk analysis
            - Use GBM to estimate a diversified portfolio upside and downside risk. 
        - Synthetic portfolio for single equity (Not working)
            - Use linear regression (tried Ridge and Lasso) to fit a portfolio of equity to a newly IPO stock.
            - Not working as expected due to insufficient historical data, I'll be back after a while.

# About me
- CUHK BSc. RMSc student
