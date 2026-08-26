# About this space
- This is the place where I save my projects' side products, you may understand my knowledge base pretty well here. 
- Request of accessing private respritory is selectively avaliable on request.


# Project Index
- Research Paper Inspired Projects _(btw, trading algos here are abandoned plans)_
    - Extreme-Value-Aware Synthetic Data Generator for Stress Testing/TCN_ARAE_EtaT.ipynb
        - TCN-Based Autoencoder: Replaces recurrent networks with 1D causal convolutions (TCNs) to process continuous log-returns, eliminating recency bias and natively preserving high-frequency, jagged market volatility
        - EVT-Driven Tail Engine ($\eta$-RNG): Generates extreme market shocks bounded by Generalized Pareto Distribution (GPD) limits, utilizing a Wasserstein penalty with a linear warmup to force realistic fat-tailed crashes and rallies without mode collapse.
        - WGAN-GP Organizer: Employs a Wasserstein GAN with a Gradient Penalty as a ranking mechanism to seamlessly evaluate and integrate the generated extreme shocks into the continuous historical market manifold.
    - Extreme-Value-Aware Synthetic Data Generator for Stress Testing/STSG_ARAE_EtaT.ipynb
        - Apply NLP autoencoder technique and extreme event RNG injection for extreme event awared time series data generation.
        - Credit: Chang, K., & Sapsis, T. P. (2025) for EVT injection, Zhao et al (2018) for autoencoder structure
    - Online Portfolio Selection/Extrema Detection Strategy
        - Multiple revision of the model all included in the folder in .ipynb format.
        - The closest to succeeded and confidential version is Turning_Point_Analysing_Engine.ipynb despite there is still some distance away from it.
        - Fusion model open up new potential but it's computational intensive and abandoned. 
        - Credit: Fang et al. (2024) for GRU model interval
    - Time Series Analysis Toolkit/Vanilla_GRU_TS_Generator.ipynb
        - Applied (η-) Learning on both tail.
        - Recursively use GRU for time series data generation.
        - Inject randomness with adding normally distributed noise and multiplying log-normally distributed shocks to force regime change.
        - Key Takeaway: Vanilla GRU can be a good for identifying momentum but discouraged to generate regime change events, stochastic injection is necessary for generating meaningful data.
        - Credit: Chang, K., & Sapsis, T. P. (2025) for EVT injection
    - Time Series Analysis Toolkit/Random_Return_Generator.py
        - Use semi-supervised deep-learning to generate extreme event awared path for stress-testing.
        - Completely adapted the original logic to time series related work.
        - Note: Foundation for deep-learning application in stress-testing, relevent code will appear in future works. 
        - Credit: Chang, K., & Sapsis, T. P. (2025) for EVT injection
    - Online Portfolio Selection/Adaptive_Mean_Reversion_Trading_Algo_(Tsang_et_al)
        - A mean reversion, robust optimization ply dynamic parameter selection model that consider transaction cost.
        - Slightly modified the parameter selection logic with EWMA and optimization logic to allow risk seeking behaviour in some regimes.
        - Credit: Tsang et al (2025) as base model
    - Online Portfolio Selection/Recursive_OLS_Algo_(Li_et_al)
        - A recursive least square prediction combined with exponential weighting FTW strategy.
        - Note: DMD is just a hype, I do not see it appear in the paper, its just recursive multiple regression with Sherman-Morrison updating, and allocated with exponential gradient
        - Credit: Li et al (2026) as base model
    - Online Portfolio Selection/Robust_Optimized_Recursive_OLS_Trading_Algo_(Hybrid)
        - Combined recursive least square prediction and robust optimization
        - Credit: Li et al (2026), Tsang et al (2025) both as base model
    - Online Portfolio Selection/Adaptive_Recursive_OLS_(Hybrid)
        - Combined recursive least square prediction, robust optimization and dynamic parameter selection with EWMA logic.
        - Note: This is absolutely not worth doing, the return is worse then constant parameter or mean reversion. 
        - Credit: Li et al (2026), Tsang et al (2025) both as base model
    - Reference list:
        - Chang, K., & Sapsis, T. P. (2025). Extreme event aware ($\eta$-) learning. arXiv preprint arXiv:2510.19161.
        - Fang, L., Chen, Y., Zhong, W., & Ma, P. (2024). Bayesian knowledge distillation: A Bayesian perspective of distillation with uncertainty quantification. In Proceedings of the 41st International Conference on Machine Learning (PMLR 235).
        - Li, Jiahao & Zhang, Yong & Zheng, Xiaoteng, 2026. "Dynamic mode decomposition for online portfolio selection task," European Journal of Operational Research, Elsevier, vol. 328(1), pages 349-365.
        - Tsang, M. Y., Sit, T., & Wong, H. Y. (2025). Adaptive robust online portfolio selection. European Journal of Operational Research, 321(1), 214–230.
        - Zhao, J., Kim, Y., Zhang, K., Rush, A. M., & LeCun, Y. (2018). Adversarially regularized autoencoders. Proceedings of the 35th International Conference on Machine Learning, PMLR 80.  

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
