import numpy as np
from nn_forward import nn_forward
from nn_backward import nn_backward
from nn_applygradient import nn_applygradient

def nn_train(nn,train_x,train_y):
    batch_size = nn.batch_size
    m = train_x.shape[0]
    num_batches = m / batch_size
    kk = np.random.permutation(m)
    for l in range(int(num_batches)):
        batch_x = train_x[kk[l * batch_size : (l + 1) * batch_size], :] #(l+1)*batch_size也可以改成max((l+1)*batch_size, len(kk))
        batch_y = train_y[kk[l * batch_size : (l + 1) * batch_size], :]
        if nn.optimization_method == 'RMSPropNesterov':
            names = ['W', 'b']
            if nn.batch_normalization:
                names += ['Gamma', 'Beta']
            # Temporarily evaluate the gradient at theta + alpha * v.
            original = {name: getattr(nn, name).copy() for name in names}
            try:
                for name in names:
                    parameters = getattr(nn, name)
                    velocity = getattr(nn, 'v' + name)
                    for k in range(nn.depth - 1):
                        parameters[k] = original[name][k] + nn.alpha * velocity[k]
                nn = nn_forward(nn, batch_x, batch_y)
                nn = nn_backward(nn, batch_y)
            finally:
                # Apply the new velocity to the original theta, even after
                # a failed forward/backward pass restore the parameters.
                for name in names:
                    getattr(nn, name).update(original[name])
        else:
            nn = nn_forward(nn,batch_x,batch_y)
            nn = nn_backward(nn,batch_y)
        nn = nn_applygradient(nn)
    return nn
