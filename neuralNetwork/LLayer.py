import numpy as np
class layer:
    def __init__(self,number_inputs,neurons,act,is_output_layer=False):
        self.output=None
        self.weights=np.random.uniform(low=0,high=1,size=(number_inputs,neurons))
        self.bias=np.random.uniform(low=0,high=1,size=(neurons))
        self.is_output_layer=is_output_layer
        self.sima=[]
        self.actvation=None
        if act == "tanh":
            self.actvation = self.tanh
            self.dertive_actvation = self.tanh_dertiv
        elif act=="sigmoid":
            self.actvation = self.sigmoid
            self.dertive_actvation = self.dertive_sgmoid
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanh_dertiv(self):
        return 1 - self.output ** 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dertive_sgmoid(self):
        return self.output * (1 - self.output)

    def forward(self,inputs,bias):
         self.output=np.dot(inputs,self.weights)
         if bias==True:
             self.output+=self.bias
         self.output=self.actvation(self.output)
