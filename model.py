"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    import numpy as np
    arr=np.array(matrix)
    return np.argmax(arr,axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    arr=np.array(matrix)
    return np.max(arr,axis=1,keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    arr=np.array(matrix)
    return np.sum(arr,axis=1,keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def row_max(matrix):
    arr=np.array(matrix)
    return np.max(arr,axis=1,keepdims=True)

def exp_shifted(logits):
    logits-=row_max(logits)
    return np.exp(logits)

# Step 5 - stable_softmax
import numpy as np

def row_max(matrix):
    arr=np.array(matrix)
    return np.max(arr,axis=1,keepdims=True)

def exp_shifted(logits):
    logits-=row_max(logits)
    return np.exp(logits)

def row_sum(matrix):
    arr=np.array(matrix)
    return np.sum(arr,axis=1,keepdims=True)

def stable_softmax(logits):
    exp_values=exp_shifted(logits)
    return exp_values/row_sum(exp_values)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    oneh=np.eye(num_classes,dtype=float)[labels]
    return oneh

# Step 7 - gather_true_class_probs
import numpy as np

def gather_true_class_probs(probs, labels):
    return probs[np.arange(len(labels)), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    true_probs = gather_true_class_probs(probs, labels)
    true_probs = np.clip(true_probs, eps, None)
    return -np.mean(np.log(true_probs))

# Step 9 - accuracy
import numpy as np
def argmax_rows(matrix):
    arr=np.array(matrix)
    return np.argmax(arr,axis=1)
def accuracy(logits_or_probs, labels):
    predicted_classes = argmax_rows(logits_or_probs)
    return np.mean(predicted_classes == labels).item()

# Step 10 - he_std
import numpy as np
def he_std(fan_in):
    std=np.sqrt(2/fan_in)
    return std.item()

# Step 11 - he_init
import numpy as np
def he_std(fan_in):
    std=np.sqrt(2/fan_in)
    return std.item()
def he_init(shape, fan_in, seed):
    np.random.seed(seed=seed)
    initm =np.random.randn(*shape)*he_std(fan_in)
    return initm

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    x=np.zeros(length,dtype=np.float64)
    return x

# Step 13 - pad_2d
def pad_2d(images, pad):
    pad_spec=((0,0),(0,0),(pad,pad),(pad,pad))
    pad_im=np.pad(images,pad_spec)
    return pad_im

# Step 14 - output_spatial_size
import numpy as np
def output_spatial_size(input_size, kernel, stride, padding):
    out=np.floor((input_size+2*padding-kernel)/stride) +1
    return int(out)

# Step 15 - im2col
import numpy as np
def pad_2d(images, pad):
    pad_spec=((0,0),(0,0),(pad,pad),(pad,pad))
    pad_im=np.pad(images,pad_spec)
    return pad_im
def output_spatial_size(input_size, kernel, stride, padding):
    out=np.floor((input_size+2*padding-kernel)/stride) +1
    return int(out)
def im2col(images, kernel_h, kernel_w, stride, padding):
    padd=pad_2d(images,padding)
    n,c,h_pd,w_pd=images.shape
    out_h=output_spatial_size(images.shape[2],kernel_h,stride,padding)
    out_w=output_spatial_size(images.shape[3],kernel_w,stride,padding)
    rows=[]
    for i in range(n):
        for j in range(out_h):
            for k in range(out_w):
                h_start=j*stride
                w_start=k*stride
                patch=padd[i,:,h_start:h_start+kernel_h,w_start:w_start+kernel_w]
                rows.append(patch.reshape(-1))
    return np.array(rows)

# Step 16 - col2im
import numpy as np
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N,C,h,w=input_shape
    out_h=output_spatial_size(h,kernel_h,stride,padding)
    out_w=output_spatial_size(w,kernel_w,stride,padding)
    image=np.zeros((N,C,h+2*padding,w+2*padding),dtype=np.float64) #tuple form
    row_idx=0
    for i in range(input_shape[0]):
        for j in range(out_h):
            for k in range(out_w):
                patch=cols[row_idx].reshape(
                    C,kernel_h,kernel_w
                )
                h_start=j*stride
                w_start=k*stride
                image[i,:,h_start:h_start+kernel_h,w_start:w_start+kernel_w]+=patch
                row_idx+=1
    if(padding>0):
        return(image[:,:,padding:-padding,padding:-padding])
    return image

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    n, c, h,w =x.shape
    c_out,c,kernel_h,kernel_w=weights.shape
    out_h=output_spatial_size(h,kernel_h,stride,padding)
    out_w=output_spatial_size(w,kernel_w,stride,padding)

    cols=im2col(x,kernel_h,kernel_w,stride,padding)

    w_c=weights.reshape(c_out,-1)
    out=cols@w_c.T+bias
    out=out.reshape(n,out_h,out_w,c_out)
    out=out.transpose(0,3,1,2)
    cache={"x_shape":x.shape, "weights":weights, "cols": cols, "kernel_h": kernel_h, "kernel_w":kernel_w,"stride":stride, "padding": padding}
    return out,cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    x_shape=cache["x_shape"]
    weights=cache["weights"]
    kernel_h=cache["kernel_h"]
    kernel_w=cache["kernel_w"]
    stride=cache["stride"]
    padding=cache["padding"]
    n,c,h,w=x_shape
    c_out=weights.shape[0]
    d_out=d_out.transpose(0, 2, 3, 1)
    d_out=d_out.reshape(-1,c_out)
    w_c=weights.reshape(c_out,-1)
    d_col=d_out@w_c
    dx=col2im(d_col,x_shape,kernel_h,kernel_w,stride, padding)
    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    weights=cache["weights"]
    cols=cache["cols"]
    kernel_h=cache["kernel_h"]
    kernel_w=cache["kernel_w"]
    c_out,c,_,_=weights.shape
    d_out_flat=d_out.transpose(0,2,3,1)
    d_out_flat=d_out_flat.reshape(-1,c_out)
    dw_col=d_out_flat.T@cols
    dw_col = dw_col.reshape(c_out, c, kernel_h, kernel_w)
    return dw_col

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    return np.sum(d_out,axis=(0,2,3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    dx=conv2d_grad_input(d_out,cache)
    dW=conv2d_grad_weights(d_out,cache)
    db=conv2d_grad_bias(d_out)
    return dx,dW,db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    n,c,h,w=x.shape
    out_h=output_spatial_size(h,kernel,stride,0)
    out_w=output_spatial_size(w,kernel,stride,0)
    out=np.zeros((n,c,out_h,out_w),dtype=np.float64)
    argmax=np.zeros((n,c,out_h,out_w),dtype=np.int64)
    for  i  in range(n):
        for j in range(c):
            for k in range(out_h):
                for l in range (out_w):
                    h_start=k*stride
                    w_start=l*stride
                    window=x[i,j,h_start:h_start+kernel,w_start:w_start+kernel]
                    flat_win=window.reshape(-1)
                    arg=np.argmax(flat_win)
                    out[i,j,k,l]=flat_win[arg]
                    argmax[i,j,k,l]=arg
    cache={"x_shape":x.shape,"argmax":argmax,"kernel":kernel,"stride":stride}
    return out, cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    window=np.zeros((kernel,kernel),dtype=np.float64)
    row=argmax_index//kernel
    col=argmax_index%kernel
    window[row,col]=grad_value
    return window

# Step 24 - maxpool2d_backward
import numpy as np
def maxpool2d_backward(d_out, cache):
    x_shape=cache["x_shape"]
    argmax=cache["argmax"]
    kernel=cache["kernel"]
    stride=cache["stride"]
    n,c,h,w=x_shape
    co,ci,out_h,out_w=d_out.shape
    dx=np.zeros((n,c,h,w))
    for i in range(n):
        for j in range(c):
            for k in range(out_h):
                for l in range(out_w):
                    h_start=k*stride
                    w_start=l*stride
                    grad_val=d_out[i,j,k,l]
                    arg=argmax[i,j,k,l]
                    grad_wind=scatter_grad_window(grad_val,arg,kernel)

                    dx[i,j,h_start:h_start+kernel,w_start:w_start+kernel]+=grad_wind
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    out=np.maximum(0,x)
    cache={"x":x}
    return out,cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    x=cache["x"]
    dx= d_out*(x>0)
    return dx

# Step 27 - flatten_forward
def flatten_forward(x):
    n,c,h,w=x.shape
    x_res=x.reshape(n,c*h*w)
    cache={"x_shape":x.shape}
    return x_res,cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    x_shape=cache["x_shape"]
    return d_out.reshape(x_shape)

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    out=x@weights+bias
    cache={"x":x,"weights":weights}
    return out,cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    weights=cache["weights"]
    dx=d_out@weights.T
    return dx

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    dw=x.T@dout
    return dw

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    return np.sum(dout,axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    dx=linear_grad_input(dout,cache)
    x= cache["x"]
    dw=linear_grad_weights(x, dout)
    db=linear_grad_bias(dout)
    return dx,dw,db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    prob=stable_softmax(logits)
    loss=cross_entropy_loss(prob,y)
    loss=float(loss)
    return 0.0 if abs(loss) < 1e-15 else loss

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    n,c=logits.shape
    probs=stable_softmax(logits)
    oh=one_hot(y,c)
    dlogits=(probs-oh)/n
    return dlogits

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    return param-lr*grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one*m+(1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    return beta_two*v+(1-beta_two)*grad**2

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    return moment / (1 - beta ** t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    m_new=adam_update_m(m,grad,beta_one)
    v_new=adam_update_v(v,grad,beta_two)
    m_hat=adam_bias_correct(m_new,beta_one,t)
    v_hat=adam_bias_correct(v_new,beta_two,t)
    params=adam_param_step(param,m_hat,v_hat,lr,eps)
    return params, m_new, v_new

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    shape=(out_channels,in_channels,kernel_size,kernel_size)
    fan_in=in_channels*kernel_size*kernel_size
    weight=he_init(shape,fan_in,seed)
    bias=init_zero_bias(out_channels)
    return {"W":weight,"b":bias}

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    shape=(in_features,out_features)
    weight=he_init(shape,in_features,seed)
    bias=init_zero_bias(out_features)
    return {'W':weight, 'b': bias}

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    conv1=init_conv_layer(6,in_channels,5,seed)
    conv2=init_conv_layer(16,6,5,seed+1)
    fc1=init_linear_layer(16*4*4,120,seed+2)
    fc2=init_linear_layer(120,num_classes,seed+3)
    return {"conv1":conv1,"conv2":conv2,"fc1":fc1,"fc2":fc2}

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    conv_out,conv_cache=conv2d_forward(x,W,b,stride,pad)
    relu_out,relu_cache=relu_forward(conv_out)
    pool_out,pool_cache= maxpool2d_forward(relu_out,pool_size,pool_size)
    cache={"conv_cache":conv_cache,"relu_cache": relu_cache,"pool_cache":pool_cache}
    return pool_out,cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    flat,flatten_cache=flatten_forward(x)
    fc1_out,fc1_cache=linear_forward(flat,fc1['W'],fc1['b'])
    relu_out,relu_cache=relu_forward(fc1_out)
    logits,fc2_cache=linear_forward(relu_out,fc2['W'],fc2['b'])
    cache={"flatten_cache":flatten_cache,"fc1_cache":fc1_cache,"relu_cache":relu_cache,"fc2_cache":fc2_cache}
    return logits,cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    out1,cache1=forward_conv_block(x,params["conv1"]["W"],
        params["conv1"]["b"],pool_size=2,stride=1,pad=0)
    out2, cache2 = forward_conv_block(out1,params["conv2"]["W"],
        params["conv2"]["b"],pool_size=2,stride=1, pad=0)
    logits, classifier_cache = forward_classifier_block(out2,params["fc1"],params["fc2"])
    caches = {"block1": cache1,"block2": cache2,"classifier": classifier_cache}

    return logits, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    conv_cache = cache['conv_cache']
    relu_cache = cache['relu_cache']
    pool_cache = cache['pool_cache']
    dpool = maxpool2d_backward(dout, pool_cache)
    drelu = relu_backward(dpool, relu_cache)
    dx, dW, db = conv2d_backward(drelu, conv_cache)
    return dx,dW,db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    flatten_cache = cache['flatten_cache']
    fc1_cache = cache['fc1_cache']
    relu_cache = cache['relu_cache']
    fc2_cache = cache['fc2_cache']
    drelu, dW_fc2, db_fc2 = linear_backward(dlogits, fc2_cache)
    dfc1 = relu_backward(drelu, relu_cache)
    dflat, dW_fc1, db_fc1 = linear_backward(dfc1, fc1_cache)
    dx = flatten_backward(dflat, flatten_cache)
    return {'dx': dx,'fc1': {'dW': dW_fc1, 'db': db_fc1},
        'fc2': {'dW': dW_fc2, 'db': db_fc2}}

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    cls_grads = backward_classifier_block(dlogits, caches['classifier'])
    dx2, dW2, db2 = backward_conv_block(cls_grads['dx'], caches['block2'])
    dx1, dW1, db1 = backward_conv_block(dx2, caches['block1'])
    return {'conv1': {'dW': dW1, 'db': db1},'conv2': {'dW': dW2, 'db': db2},
        'fc1': cls_grads['fc1'],'fc2': cls_grads['fc2']}

# Step 51 - lenet_predict
def lenet_predict(x, params):
    logits, caches= lenet_forward(x, params)
    return argmax_rows(logits).astype(int)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    rng=np.random.default_rng(seed)
    y=rng.integers(0, num_classes, size=num_samples)
    x=rng.standard_normal((num_samples, in_channels, image_size, image_size))
    shift=(y - (num_classes - 1) / 2)
    shift=shift.reshape(-1, 1, 1, 1)
    x+=shift
    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    np.random.seed(seed)
    return np.random.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    n=x.shape[0]
    indices=shuffle_indices(n, seed=seed)
    test_size=int(n*test_fraction)
    test_idx=indices[:test_size]
    train_idx=indices[test_size:]
    x_train, y_train=x[train_idx], y[train_idx]
    x_test, y_test=x[test_idx], y[test_idx]
    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    n=x.shape[0]
    indices = shuffle_indices(n, seed=seed)
    num_batches=n//batch_size
    for i in range(num_batches):
        start=i*batch_size
        end=start+batch_size
        batch_idx=indices[start:end]
        yield x[batch_idx], y[batch_idx]

# Step 56 - train_step
def train_step(params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step):
    logits, caches=lenet_forward(xb, params)
    loss=softmax_cross_entropy_forward(logits, yb)
    dlogits=softmax_cross_entropy_backward(logits, yb)
    grads=lenet_backward(dlogits, caches)
    new_params={}
    new_opt_state={}
    for lyr_name in params:        
        new_params[lyr_name]={}
        new_opt_state[lyr_name]={}
        for param_name in params[lyr_name]:   
            p=params[lyr_name][param_name]
            g=grads[lyr_name]['d'+ param_name] 
            m=opt_state[lyr_name][param_name]['m']
            v=opt_state[lyr_name][param_name]['v']
            p_new, m_new, v_new =adam_step(p, g, m, v, step, lr, beta_one, beta_two, eps)
            new_params[lyr_name][param_name] = p_new
            new_opt_state[lyr_name][param_name] = {'m': m_new,'v': v_new}
    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size, lr, beta_one, beta_two, eps, step_counter, seed=0):
    losses=[]
    for xb,yb in iterate_minibatches(x, y, batch_size, seed=seed):
        step_counter += 1
        params, opt_state, loss = train_step(params, opt_state, xb, yb,lr, beta_one, beta_two, eps, step_counter)
        losses.append(float(loss))
    return params, opt_state, step_counter, losses

# Step 58 - train_loop
def train_loop(params, x_train, y_train, num_epochs, batch_size, lr=1e-3, beta_one=0.9, beta_two=0.999, eps=1e-8, seed=0):
    opt_state={}
    for layer_name in params:
        opt_state[layer_name]={}
        for param_name in params[layer_name]:
            shape = params[layer_name][param_name].shape
            opt_state[layer_name][param_name]={'m': np.zeros(shape),'v': np.zeros(shape)}
    step_counter=0
    all_losses=[]
    for epoch in range(num_epochs):
        params, opt_state, step_counter, epoch_losses= train_one_epoch(
            params, opt_state, x_train, y_train, batch_size,lr, beta_one, beta_two,
            eps, step_counter,seed=seed + epoch)
        all_losses.extend(epoch_losses)
    return params, all_losses

# Step 59 - evaluate
def evaluate(params, x, y):
    predi= lenet_predict(x, params)
    return float(np.mean(predi== y))

