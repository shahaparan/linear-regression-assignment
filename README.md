
Medium URL - https://shahporan2014.medium.com/practice-lab-linear-regression-0b05df90304c


# 1. Linear Regression Cost Function

Let’s, explain this code, Here is compute_cost function which takes in four parameters: `x, y, w, and b`

`x` is the Input to the model (Population of cities)

`y` is the Label (Actual profits for the cities).

`w and b` are the parameters of the model.

Return — total_cost (float): The cost of using `w,b` as the parameters for linear regression to fit the data points in ` x and y.`

Predication of the model — `f_wb(xi) = w * xi + b`

the cost — `cost_i = (f_wb(xi) — yi)²`

Total Cost over all examples — `J(w, b) = (1/2m) * sum(cost_i)`


## Code implementation 

```python
# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """

    # number of training examples
    m = x.shape[0] 
    
    total_cost = 0
    
    ### START CODE HERE ###
    for i in range(m):                                
        f_wb_i = np.dot(x[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        
        total_cost = total_cost + (f_wb_i - y[i]) **2       #scalar
        
    total_cost = total_cost / (2 * m)                      #scalar   
    ### END CODE HERE ### 

    return total_cost
   ```

# 2. Compute the cost function for linear regression.

```python
# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    
    # Number of training examples
    m = x.shape[0]
    
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    for i in range(m):
        
        # Compute prediction model
        fw_b = w * x[i] + b
        
        # Compute gradient parameters w, b
        dj_dw_i = (fw_b - y[i]) * x[i]
        dj_db_i = fw_b - y[i]
        
        # update the total gradient
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    
    # Return average gradient update
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db
```


`x (ndarray): Shape (m,)` Input to the model (Population of cities)

`y (ndarray): Shape (m,)` Label (Actual profits for the cities)

`w, b (scalar)`: Parameters of the model

Returns

`dj_dw (scalar)`: The gradient of the cost w.r.t. the parameters w

`dj_db (scalar)`: The gradient of the cost w.r.t. the parameter b

prediction of the model `f_wb = w * X[i] + b`

gradient for the parameter `b ∂J(w,b)/∂b^(i) = f_wb - y[i]`

gradient for the parameter `w ∂J(w,b)/∂w^(i) = (f_wb - y[i]) * X[i]`

total gradient update for the parameters `w, b ∂J(w,b)/∂b = (1/m) * ∑(∂J(w,b)/∂b^(i)) ∂J(w,b)/∂w = (1/m) * ∑(∂J(w,b)/∂w^(i))`
