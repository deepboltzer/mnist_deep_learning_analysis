# mnist_deep_learning_analysis
Deep Learning MNIST with 3 layer mlps

Analysis results: 

1. test and validation:
   - SGD with nesterov converges the fastest w.r.t to test and training loss. 
   - For the validation accuracy SGD with nesterov and SGD with nesterov and l2 regularization got the best training accuracy. 
     We see that optimization with nesterov is working well on mnist dataset. 
   - SGD with nesterov and l1 regularization performs not so well and semms to need more than 20 epochs. But also the training loss   
     falls very slow for this method. 
     
2. classification accuracy: 
   - Classification accuracy for MLP-I:   95.39999999999999
   - Classification accuracy for MLP-II:  97.59
   - Classification accuracy for MLP-III: 87.74
   - Classification accuracy for MLP-IV:  97.25
   
   - For the classification accuracy SGD with nesterov and SGD with nesterov and L2 regularization are close together. What we see is that nesterov optimization on the weights can be very helpful on mnist dataset.
   
   - Open Question: What happens if we use more than 20 epochs? 
    
