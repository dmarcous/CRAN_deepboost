#Deepboost modeling.

Provides deepboost models training, evaluation, predicting and hyper parameter optimising using grid search and cross validation.

##Details

Based on Google's Deep Boosting algorithm by Cortes et al.

See [this paper](http://www.cs.princeton.edu/~usyed/CortesMohriSyedICML2014.pdf) for details

Adapted from Google's C++ deepbbost implementation :

<https://github.com/google/deepboost>

Another version for the package that uses the original unmodified algorith exists in :

<https://github.com/dmarcous/deepboost>

##Installation

From CRAN : 

    install.packages("deepboost")

##Examples

Choosing parameters for a deepboost model :

    best_params <- deepboost.gridSearch(formula, data)

Training a deepboost model :

    boost <- deepboost(formula, data,
                        num_iter = best_params[2][[1]], 
                        beta = best_params[3][[1]], 
                        lambda = best_params[4][[1]], 
                        loss_type = best_params[5][[1]]
                        )

Print trained model evaluation statistics :                         

    print(boost)

Classifying using a trained deepboost model :

    labels <- predict(boost, newdata)
    
See Help / demo directory for advanced usage.

##Credits

R Package written and maintained by :

Daniel Marcous <dmarcous@gmail.com>

Yotam Sandbank <yotamsandbank@gmail.com>
