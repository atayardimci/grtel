# Graph-Regularized Tensor Ensemble Learning for Financial Forecasting

Financial machine learning poses significant challenges due to its large-dimensional, irregular, and multi-modal data with exceedingly low signal-to-noise ratio. Conventional machine learning methods are ill-equipped to tackle the challenges of modern financial big data, encountering issues with computational complexity and learning generalization.

This project explores a solution to these challenges using modern signal processing techniques based on tensors, tensor decompositions, and graphs. By pursuing a graph-theoretic approach, specifically portfolio cuts, to cluster assets and by using resulting clusters and notions such as degree centrality, we tensorize financial data samples along specific modes. This introduces a rigorous method for tensorizing multi-dimensional data acquired on irregular domains and equip the resulting tensor representation with more meaning, accounting for the structural dependencies within data, which can later be utilized with tensor decompositions.

The tensorized samples are then integrated into a Tensor Ensemble Learning (TEL) framework for classifying stock price movements. By benefiting from the ensemble learning abilities of the framework, while exploiting the multi-dimensional structure of the input samples by virtue of tensors and their decompositions, we propose a new framework called Graph-Regularized Tensor Ensemble Learning (GRTEL) and demonstrate its benefits through numerical simulations based on real-world financial data.

Note: Reimplentation of my Imperial EEE MEng project. Original resources available at https://github.com/atayardimci/GR-TEL-for-Financial-Forecasting.
