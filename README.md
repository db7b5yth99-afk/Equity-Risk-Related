# About this space
- This is the place where I save my Python learning projects, you may understand my knowledge base pretty well here. 
- Code here are useful as I may use them to feed LLM for localization when doing simular task. 
- Request of accessing private respritory is selectively avaliable on request.


# Project Index
- Research Paper Model Recreation & Modifying _(OLPS related topic built on frictionless assumption)_
    - Time Series Analysis Toolkit/Vanilla_GRU_TS_Generator.ipynb
        - Applied η- Learning on both tail.
        - Recursively use GRU for time series data generation.
        - Inject randomness with adding normally distributed noise and multiplying log-normally distributed shocks to force regime change.
        - Key Takeaway: Vanilla GRU can be a good for identifying momentum but discouraged to generate regime change events, stochastic injection is necessary for generating meaningful data.
    - Time Series Analysis Toolkit/Extreme Event Aware (η-) Learning (Chang & Sapsis, 2025) inspired Time Series Sample Generator
        - Use semi-supervised deeplearning to generate extreme event awared path for stress-testing.
        - Completely adapted the original logic to time series related work.
        - Note: Very Useful for stress-testing. 
        - Credit: Chang, K., & Sapsis, T. P. (2025)
    - Online Portfolio Selection/Adaptive_Mean_Reversion_Trading_Algo_(Tsang_et_al)
        - A mean reversion, robust optimization ply dynamic parameter selection model that consider transaction cost.
        - Slightly modified the parameter selection logic with EWMA and optimization logic to allow risk seeking behaviour in some regimes.
        - Credit: Tsang et al (2025)
    - Online Portfolio Selection/Recursive_OLS_Algo_(Li_et_al)
        - A recursive least square prediction combined with exponantial weighting FTW strategy.
        - Note: DMD is just a hype, I do not see it appear in the paper, its just recursive multiple regression with Sherman-Morrison updating, and allocated with exponential gradient
        - Credit: Li et al (2026)
    - Online Portfolio Selection/Robust_Optimized_Recursive_OLS_Trading_Algo_(Hybrid)
        - Combined recursive least square prediction and robust optimization
        - Credit: Li et al (2026), Tsang et al (2025)
    - Online Portfolio Selection/Adaptive_Recursive_OLS_(Hybrid)
        - Combined recursive least square prediction, robust optimization and dynamic parameter selection with EWMA logic.
        - Note: This is absolutely not worth doing, the return is worse then constant parameter or mean reversion. 
        - Credit: Li et al (2026), Tsang et al (2025)
    - Referance list:
        - Chang, K., & Sapsis, T. P. (2025). Extreme event aware ($\eta$-) learning. arXiv preprint arXiv:2510.19161.
        - Li, Jiahao & Zhang, Yong & Zheng, Xiaoteng, 2026. "Dynamic mode decomposition for online portfolio selection task," European Journal of Operational Research, Elsevier, vol. 328(1), pages 349-365.
        - Tsang, M. Y., Sit, T., & Wong, H. Y. (2025). Adaptive robust online portfolio selection. European Journal of Operational Research, 321(1), 214–230.

- Other Projects (From Latest to Oldest):
    - Online Portfolio Selection
        - xgboost_discrete
            - Use RSI, FS, MA to create discrete signal, feeding to xgboost for regime prediction.
            - Note: The result is dominated by a hybrid deep-learning plus bayesian approach archived in private respiratory.
        - data_ingestion
            - Utility for backtest data feeding.
        - performance_analysis
            - Utility for backtest validation.
    - Time series analysis toolkit
        - A DMD Approach for Index Prediction
            - Applied Dynamic Mode Decomposition to predict S&P 500 1-day Close based on behaviour of different concept ETFs. 
        - A Dynamic FFT Implementation on TSA
            - Build a model that dynamically apply FFT in different window at a list of specified period to better decompose the time series cycles.
            - Include Gaussian fade-in and out option to simulate reallife financal time series pereformance. 
        - Economic cycle indicator
            - Build indicator based on 20y historical data using fast fourier transform. 
        - Single equity analysis
            - Use SVJ model alongside GARCH(1,1) to visualize upside and downside risk of single stock including volatile microcaps.
        - Portfolio risk analysis
            - Use GBM to estimate a diversified portfolio upside and downside risk. 


# About me
- CUHK BSc. RMSc student
