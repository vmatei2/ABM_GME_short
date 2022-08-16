This zipped folder contains all code used in generating the outcomes presented in the thesis.

The main file that generates results is SimulationClass.

Supporting files are as follows:

- RedditTrader, the base class for InfluentialRedditTrader and RegularRedditTrader
- InstitutionalInvestor, class modelling the behaviour of hedge funds throughout the event
- RedditInvestorTypes.py, file containing the definition of an enum, used as an attribute of the RedditTrader
- ParameterEstimation.py
- SensitivityAnalysis.py, used in analysis the results of sensitivity analysis - the code to obtain sensitivity analysis results is found within SimulationClass.py
- MarketEnviornment, class modelling the artificial environment through which the gent classes interact, as described in the thesis
- helpers folder with the different helpers file
    - extract_reddit_data.py was used in the empirical analysis of the Kaggle data retrieved on
    the development of the wsb community throughout the event
    - the kaggleData files are under the kaggleData folder
    
- tests folder, used in ensuring calculation functions behave as expected


The zip represents the main branch used in development. A separate *dynamic_hedging* branch 
which has the code to support the market-maker implementation presented
in the Appendix of the thesis has also been created.

