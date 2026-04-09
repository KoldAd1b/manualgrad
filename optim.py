class SGD: 
    def __init__(self,params,lr=0.01):
        self.parameters = params
        self.lr = lr
    
    def step(self):
        for param in self.parameters:
            assert param.grad.shape == param.params.shape , "Gradient and parameter shapes do not match"
            param.params -= self.lr * param.grad
    
    def zero_grad(self):
        for param in self.parameters:
            param._zero_grad()


class AdamOptimizer: 

    def __init__(self,params,lr=0.001,betas=(0.9,0.999),eps=1e-8):
        self.parameters = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [cp.zeros_like(param.params) for param in params]
        self.v = [cp.zeros_like(param.params) for param in params]
    
    def step(self):
        self.t += 1
        for i,param in enumerate(self.parameters):
            assert param.grad.shape == param.params.shape , "Gradient and parameter shapes do not match"
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** 2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            param.params -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.parameters:
            param._zero_grad()