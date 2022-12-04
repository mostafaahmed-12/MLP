import Neural_Network as nn
hidden_layers=2
neurons=[3,4]
learing_rate=0.1
epochs=600
actv="sigmoid"
bias=True


nn.run(hidden_layers,learing_rate,neurons,epochs,actv,bias)
