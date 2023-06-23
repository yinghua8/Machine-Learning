# Machine-Learning
### HW1
This HW is mainly designed to teach us the mathematical concept of linear regression. Following are the four demands for Part I. Linear regression:

1. Please plot the data points(only the Training Set) and the fitting curve for M = 1,3,5,10,20 and 30, respectively.
2. Please plot the Mean Square Error evaluated on the Training Set and the Testing Set separately for M from 1 to 30. 
3. Please apply the 5-fold cross-validation in your training stage to select the best order M and evaluate the mean square error on the Testing Set. Plot the fitting curve and data points (only the Training Set). You should briefly express how you select the best order M step-by-step.
4. Considering regularization, please use the modified error function $\tilde E(w) = \frac{1}{2}\Sigma_{i=1}^N{y(x_i, w) - t_i}^2 + \frac{\lambda}{2}||w||^2$ where $||w||^2 = w_1^2 + w_2^2 + ... + w_M^2$. Repeat Part I -1. and Part I-2. with $\lambda = 1/10$. (You can also try to change the value of λ and discuss what happens under different λ values.)

Part II. Bayesian Linear Regression asks to use the Training Set in Part I, apply the sigmoidal basis functions in Part I with M=10, and implement Bayesian linear regression. In order to discuss how the amount of training data affects the regression process, please implement a sequential estimation: Please sequentially compute the mean $m_N$ and the covariance matrix $S_N$ for the posterior distribution $p(w|t) = N(w|m_N, S_N)$ with the given prior $p(w) = N(w|m_0, S_0)$ with $m_0 = 0, S_0 = 10^{-6}I$. The predictive distribution $p(t|X, w, \beta)$ is chosen to be $\beta = 1$. Similar to the following figures, please plot the curve of the posterior mean versus x and the region spanning one standard deviation on either side of the mean curve for N = 1, 2, 3, 4, 5, 10, 20, 30, 40, 50.

And the image below shows the different functions used in this HW.

![image](https://github.com/yinghua8/Machine-Learning/assets/71891722/b9e32fb4-12b7-4c54-80e2-91395ceddaf6)

### HW2
This HW is mainly designed to teach us the mathematical concept of classification. The statement of Part I. is: Consider that there are a group of college basketball players, who come from four different high schools. Now we analyze the Performance-Rating of “skill” and the “athleticism” for every player, as shown in Figure 1, where the blue, green, red, and black dots represent the players from High school 1, High school 2, High school 3, and High school 4, respectively. Noted that there are 400, 250, 200, and 150 players from High school 1, High school 2, High school 3, and High school 4, respectively. Please implement the algorithms of the generative model and the discriminative model to classify the data and plot the corresponding decision boundaries.

![image](https://github.com/yinghua8/Machine-Learning/assets/71891722/cf43643b-ce81-4b40-ab65-c12e46270d02)

Part II. According to the overall Performance-Rating of “skill” and the “athleticism” for each high school in Figure 1, we classify High School 2 to be Class A, High School 1 and High School 4 to be Class B, and High School 3 to be Class C, as illustrated in green, blue, and red dots, respectively in Figure 3.

![image](https://github.com/yinghua8/Machine-Learning/assets/71891722/c550c85e-3c12-4e12-9fa9-4e8f95f87c92)


### HW3
This HW is mainly designed to teach us how to run the functions of a Support Vector Machine. In this homework, we need to use SVM models to classify the handwritten numbers from zero to four.
To do:
1. Implement ν-SVM model: compare the performance difference with the different kernel types and different ν values. 
2. Implement C-SVM model: compare the performance difference with the different kernel types and different C values. 
3. Choose one of the models you’ve trained to do analysis and find those “supporting vectors” for this model.

### HW4
This HW is mainly designed to teach us to run a structure of CNN. Use the CNN model to do the classification of the animal pictures, which are classified into 10 different categories, including dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, and elephant.
