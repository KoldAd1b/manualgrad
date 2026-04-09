import cupy as cp
import numpy as np
import math
import os

##### QUICK ENABLE FOR TENSOR CORE OPS (for cupy only) ###
device = cp.cuda.Device()
cc_major, cc_minor = device.compute_capability
if int(cc_major) >= 8:
    os.environ["CUPY_TF32"] = "1"
##########################################


class Operation: 
    def forward(self,x):
        raise NotImplementedError
    def backward(self,grad):
        raise NotImplementedError
    
class GradTensor:
    def __init__(self,params):
        self.params = params
        self.shape = params.shape
        with cp.cuda.Device(params.device):
            self.grad = cp.zeros_like(params)

    def _zero_grad(self):
        with cp.cuda.Device(self.params.device):
            self.grad = cp.zeros_like(self.params)

    
class GradLayer:

    def parameters(self):

        params = []
        for attr,value in self.__dict__.items():
            if isinstance(value,GradTensor):
                params.append(value)
            elif isinstance(value,GradLayer):
                params.extend(value.parameters())
        return params
    

class Linear(GradLayer):
    
    def __init__(self,in_features,out_features,bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = GradTensor(cp.random.normal(scale=0.02,size=(in_features,out_features),dtype=cp.float32))
        if bias:
            self.bias = GradTensor(cp.zeros((out_features,), dtype=cp.float32))
        else:
            self.bias = None
            

    def forward(self,x):
        self.x = x
        output = cp.matmul(x,self.weight.params)
        if self.bias is not None:
            output += self.bias.params
        return output   

    def backward(self,grad_output):
        self.weight.grad += cp.matmul(self.x.T,grad_output)
        if self.bias is not None:
            self.bias.grad += cp.sum(grad_output,axis=0)
        return cp.matmul(grad_output,self.weight.params.T)

class SLOW_SoftMax(Operation):
    """
    Softmax normalization to convert logits -> probabilities
    Backward explicitly computes the jacobian for each sample
    """
    def forward(self, x):
        self.x = x
        shifted_x = x - cp.max(x, axis=-1, keepdims=True)
        exp_x = cp.exp(shifted_x)
        self.probs = exp_x / cp.sum(exp_x, axis=-1, keepdims=True)
        return self.probs

    def backward(self, output_grad):
        gradient = cp.zeros_like(self.probs)
        batch_size = len(self.x)
        for i in range(batch_size):
            sample_probs = self.probs[i]
            j = -sample_probs.reshape(-1,1) * sample_probs.reshape(1,-1)
            j[cp.diag_indices(j.shape[0])] = sample_probs * (1 - sample_probs)
            a = output_grad[i] @ j
            gradient[i] = a
       
        return gradient

class SoftmaxOperation(Operation):
    def forward(self,x):
        self.x = x
        shifted_x = x - cp.max(x,axis=-1,keepdims=True)
        exps = cp.exp(shifted_x)
        self.props = exps / cp.sum(exps,axis=-1,keepdims=True)
        return self.props
    
    def backward(self,grad_output):
        dot_product = cp.sum(grad_output * self.props,axis=-1,keepdims=True)
        return self.props * (grad_output - dot_product)


class ReluOperation(Operation):
    def forward(self,x):
        self.x = x
        return cp.clip(x, a_min=0.0, a_max=None)
    
    def backward(self,grad_output):
        return grad_output * (self.x > 0).astype(cp.float32)

class Embedding(GradLayer):

    def __init__(self,vocab_size,embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = GradTensor(cp.random.normal(scale=0.02,size=(vocab_size,embed_dim),dtype=cp.float32))

    def forward(self,x):
        self.x = x
        return self.embedding.params[x]
    
    def backward(self,grad_output):
        cp.add.at(self.embedding.grad, self.x, grad_output)
        return None

class MSELoss(Operation):
    def forward(self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return cp.mean((y_pred - y_true)**2)
    
    def backward(self,grad_output):
        return 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]

class PositionalEmbeddings(GradLayer):

    def __init__(self,max_len,embed_dim):
        self.max_len = max_len
        self.embed_dim = embed_dim  
        self.weight = GradTensor(cp.random.normal(scale=0.02,size=(max_len,embed_dim),dtype=cp.float32))

    def forward(self,x):
        self.seq_len = x.shape[1]
        assert self.seq_len <= self.max_len, "Sequence length exceeds max length"
        self.x = x 
        return x + self.weight.params[:self.seq_len][None,:,:]

    def backward(self,grad_output):
        cp.add.at(self.weight.grad, cp.arange(self.seq_len), grad_output.sum(axis=0))
        return grad_output

class LayerNorm(GradLayer):
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = GradTensor(cp.ones(self.num_features, dtype=cp.float32))
        self.beta = GradTensor(cp.zeros(self.num_features, dtype=cp.float32))

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input of shape (B, S, E)
        Returns:
            out: normalized and affine-transformed output
        """
        self.x = x  # save input
   
        # Compute mean and variance across features (axis=-1)
        self.mean = cp.mean(x, axis=-1, keepdims=True)   # shape (B, S, 1)
        self.var  = cp.var(x, axis=-1, keepdims=True)    # shape (B, S, 1)

        # Normalize
        self.x_hat = (x - self.mean) / cp.sqrt(self.var + self.eps)

        # Scale and shift
        out = self.gamma.params * self.x_hat + self.beta.params

        return out
    
    def backward(self, output_grad):

        # Gradient w.r.t gamma and beta (affine transform)
        self.gamma.grad += cp.sum(output_grad * self.x_hat, axis=(0, 1))
        self.beta.grad  += cp.sum(output_grad, axis=(0, 1))              

        # Gradient w.r.t normalized input
        dx_hat = output_grad * self.gamma.params  # (B, S, E)
        var_eps = self.var + self.eps

        # LayerNorm backward formula (vectorized)
        mean1 = cp.mean(dx_hat, axis=-1, keepdims=True)
        mean2 = cp.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)
        dx = (dx_hat - mean1 - self.x_hat * mean2) / cp.sqrt(var_eps)

        return dx
   
class MultiHeadAttention(GradLayer):
    """
    Multi-Head Self-Attention.
    """
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = Linear(self.embed_dim, self.embed_dim)
        self.k_linear = Linear(self.embed_dim, self.embed_dim)
        self.v_linear = Linear(self.embed_dim, self.embed_dim)
        self.out_linear = Linear(self.embed_dim, self.embed_dim)
        self.softmax = SoftmaxOperation()

    def forward(self, x, attention_mask=None):
        """
        For self-attention, query = key = value = x (batch_size, seq_len, embed_dim)
        mask: optional (batch_size, 1, seq_len, seq_len) with -inf where masked, 0 otherwise.
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for linear layers: (batch_size * seq_len, embed_dim)
        x_flat = x.reshape(batch_size * seq_len, self.embed_dim)

        # Linear projections
        q = self.q_linear.forward(x_flat)
        k = self.k_linear.forward(x_flat)
        v = self.v_linear.forward(x_flat)
        
        # Reshape back to (batch_size, seq_len, embed_dim) then to heads
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = cp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        ### Mask out any non causal positions ###
        self.attention_mask = attention_mask
        if self.attention_mask is not None:
            scores += self.attention_mask
        
        # Apply softmax
        scores_reshaped = scores.reshape(batch_size * self.num_heads, seq_len, seq_len)
        probs = self.softmax.forward(scores_reshaped)
        probs = probs.reshape(batch_size, self.num_heads, seq_len, seq_len)
        
        self.probs = probs
        self.q = q
        self.k = k
        self.v = v
        
        # Attention output
        attn = cp.matmul(probs, v)
        
        # Concat heads: (batch, seq, embed_dim)
        attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Reshape for out_linear
        attn_flat = attn.reshape(batch_size * seq_len, self.embed_dim)
        out = self.out_linear.forward(attn_flat)
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        
        return out

    def backward(self, output_grad):
        """
        Backward through multi-head attention.
        Returns grad w.r.t. query, but since self-attn assumes query=key=value, we sum gradients.
        """
        batch_size, seq_len, _ = output_grad.shape
        
        # Back through out_linear
        output_grad_flat = output_grad.reshape(batch_size * seq_len, self.embed_dim)
        grad_attn_flat = self.out_linear.backward(output_grad_flat)
        grad_attn = grad_attn_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        ### Backward through attn = probs @ v ###
        ### If Y = XW, dL/dW = X^T(dL/dY) and dL/dX = (dL/dY)W^T ###
        ### This was how our linear layer worked, the same idea applied here ###
        grad_probs = cp.matmul(grad_attn, self.v.transpose(0, 1, 3, 2))
        grad_v = cp.matmul(self.probs.transpose(0, 1, 3, 2), grad_attn)
        
        # Back through softmax
        grad_probs_reshaped = grad_probs.reshape(batch_size * self.num_heads, seq_len, seq_len)
        grad_scores_reshaped = self.softmax.backward(grad_probs_reshaped)
        grad_scores = grad_scores_reshaped.reshape(batch_size, self.num_heads, seq_len, seq_len)
        
        ### 0 out grads from non causal positions ###
        if self.attention_mask is not None:
            grad_scores = cp.where(self.attention_mask==-np.inf, 0, grad_scores)

        # Back through scaling
        grad_scores /= math.sqrt(self.head_dim)
        
        ### Backward through scores = q @ k.T ###
        ### Just like before lets first do dL/dQ = dL/dS (k^T)^T = dL/dS (k) ###
        grad_q = cp.matmul(grad_scores, self.k)
        ### dL/dK^T = Q^T dL/dS, but we need in terms of dL/dK to continue backprop ###
        ### to get our shapes correct. So dL/dK = [Q^T dL/dS]^T = (dL/dS)^T Q ###
        grad_k = cp.matmul(grad_scores.transpose(0, 1, 3, 2), self.q)
        
        # Transpose back
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Reshape for linear layers
        grad_q_flat = grad_q.reshape(batch_size * seq_len, self.embed_dim)
        grad_k_flat = grad_k.reshape(batch_size * seq_len, self.embed_dim)
        grad_v_flat = grad_v.reshape(batch_size * seq_len, self.embed_dim)
        
        # Back through linears
        grad_query = self.q_linear.backward(grad_q_flat)
        grad_key = self.k_linear.backward(grad_k_flat)
        grad_value = self.v_linear.backward(grad_v_flat)
        
        # Reshape back to (batch_size, seq_len, embed_dim)
        grad_query = grad_query.reshape(batch_size, seq_len, self.embed_dim)
        grad_key = grad_key.reshape(batch_size, seq_len, self.embed_dim)
        grad_value = grad_value.reshape(batch_size, seq_len, self.embed_dim)
        
        return grad_query + grad_key + grad_value

        

class FFN(GradLayer):
    def __init__(self, embed_dim, dim_feedforward):
        self.linear1 = Linear(embed_dim, embed_dim * dim_feedforward)
        self.linear2 = Linear(embed_dim * dim_feedforward, embed_dim)
        self.relu = ReluOperation()

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, embed_dim)
        x_flat = self.linear1.forward(x_flat)
        x_flat = self.relu.forward(x_flat)
        x_flat = self.linear2.forward(x_flat)
        x = x_flat.reshape(batch_size, seq_len, embed_dim)
        return x

    def backward(self, grad_output):
        batch_size,seq_len,embed_dim = grad_output.shape
        grad_output_flat = grad_output.reshape(batch_size * seq_len, embed_dim)
        grad_output_flat = self.linear2.backward(grad_output_flat)
        grad_output_flat = self.relu.backward(grad_output_flat)
        grad_output_flat = self.linear1.backward(grad_output_flat)
        grad_output = grad_output_flat.reshape(batch_size, seq_len, embed_dim)
        return grad_output

class Dropout(Operation):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # By default, layers are in training mode

    def forward(self, x):
        if self.training:
            # Create mask of zeros and ones
            self.mask = (cp.random.rand(*x.shape) >= self.p) 
            ### Scaling so mask divides all non-masked values by 1/p to maintain
            ### the overall variance of the tensor 
            self.mask = self.mask / (1.0 - self.p)
            self.mask = self.mask.astype(cp.float32)

            return x * self.mask
        else:
            return x  # No dropout in evaluation

    def backward(self, output_grad):
        if self.training:
            return output_grad * self.mask 
        else:
            return output_grad
        
    def __repr__(self):
        return f"Dropout(p={self.p})"

class TransformerBlock(GradLayer):
    def __init__(self, embed_dim, num_heads, dropout_p, dim_mult):

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.dim_mult = dim_mult

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FFN(embed_dim, dim_mult)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)

    def forward(self, x, attention_mask=None):

        ### Attention + Residual and LayerNorm ###
        attn = self.attention.forward(x, attention_mask)
        attn = self.dropout1.forward(attn)
        x = x + attn
        x = self.norm1.forward(x)

        ### Feedforward + Residual and LayerNorm ###
        ff_out = self.ff.forward(x)
        ff_out = self.dropout2.forward(ff_out)
        x = x + ff_out
        x = self.norm2.forward(x)

        return x
    
    def backward(self, output_grad):

        ### Backward through LayerNorm2 (This is our grad for residual) ###
        grad = self.norm2.backward(output_grad)

        ### Backward through Dropout2 ###
        grad_drop = self.dropout2.backward(grad)

        ### Backward through FeedForward ###
        grad_ff = self.ff.backward(grad_drop)

        ### Add Residual Gradient ###
        grad = grad + grad_ff

        ### Backward through Layernorm1 (This is our grad for residual) ###
        grad = self.norm1.backward(grad)

        ### Backward through Dropout1 ###
        grad_drop = self.dropout1.backward(grad)

        ### Backward through Attention ###
        grad_attn = self.attention.backward(grad_drop)

        ### Residual Connection ###
        grad = grad + grad_attn

        return grad
    
    def __repr__(self):
        return f"TransformerBlock(embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout_p={self.dropout_p}, dim_mult={self.dim_mult})"


class SLOW_CrossEntropyLoss(Operation):
    """
    Cross Entropy Loss that assumes inputs are probabilities (already softmaxed).
    If we are passing in probs, we will have to backprop though softmax later!
    Targets are integer class indices (no one-hot).
    """
    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size,) integer class indices
        y_pred: (batch_size, num_classes) probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred
        batch_size = y_pred.shape[0]
        # Pick probability of the correct class for each sample
        correct_class_probs = y_pred[cp.arange(batch_size), y_true]
        # Compute loss
        loss = -cp.mean(cp.log(correct_class_probs + 1e-12))  # epsilon for safety
        return loss

    def backward(self):
        """
        Gradient wrt probabilities
        dL/dy_pred = -y_true / y_pred
        But since y_true is an index, we only subtract at the correct class.
        """
        batch_size, num_classes = self.y_pred.shape
        grad = cp.zeros_like(self.y_pred)
        # Only the correct class contributes to loss: -y_i / p_c
        grad[cp.arange(batch_size), self.y_true] = -1 / (self.y_pred[cp.arange(batch_size), self.y_true] + 1e-12)
        return grad
    
    def __repr__(self):
        return "SLOW_CrossEntropyLoss()"

class CrossEntropyLoss(Operation):
    """
    This loss expects raw logits. This has the benefit that we no longer need to 
    backprop though softmax anymore as we can handle the operation here!
    """
    def forward(self, y_true, logits):
        """
        y_true: [batch_size] integer class indices
        logits: [batch_size, num_classes] raw scores (NOT softmaxed)
        """
        self.y_true = y_true
        self.logits = logits
        # numerically stable softmax
        shifted_logits = logits - cp.max(logits, axis=1, keepdims=True)
        exp_logits = cp.exp(shifted_logits)
        self.probs = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)
        # pick the probabilities of the correct class for each sample
        batch_size = y_true.shape[0]
        correct_class_probs = self.probs[cp.arange(batch_size), y_true]
        # cross-entropy loss
        loss = -cp.mean(cp.log(correct_class_probs + 1e-12))
        return loss

    def backward(self):
        """
        Gradient of loss wrt logits (softmax + cross entropy simplified)
        """
        batch_size, num_classes = self.logits.shape
        grad = self.probs.copy()
        # subtract 1 at the correct class index
        grad[cp.arange(batch_size), self.y_true] -= 1

        return grad / batch_size


    
    
    def __repr__(self):
        return "CrossEntropyLoss()"
        

class FlattenForLLM(Operation):
    def forward(self, x):
        self.input_shape = x.shape
        batch_size, seq_len, embed_dim = self.input_shape
        return x.reshape(batch_size*seq_len, embed_dim)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def __repr__(self):
        return "FlattenForLLM()"


class NeuralNetwork:
    """
    The most basic Neural Network Ever!
    """
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input, attention_mask=None):
        for layer in self.layers:
            if isinstance(layer, TransformerBlock):
                input = layer.forward(input, attention_mask)
            else:
                input = layer.forward(input)
        return input

    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
        return output_grad

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, GradLayer):
                layer_parameters = layer.parameters()
                params.extend(layer_parameters)
        return params
    
    def train(self):
        """Set network to training mode."""
        self.training = True
        for layer in self.layers:
            self._set_mode_recursive(layer, True)

    def eval(self):
        """Set network to evaluation mode."""
        self.training = False
        for layer in self.layers:
            self._set_mode_recursive(layer, False)

    def _set_mode_recursive(self, layer, mode):
        if hasattr(layer, "training"):
            layer.training = mode
        if isinstance(layer, (GradLayer, Operation)):
            for attr_name, attr_value in layer.__dict__.items():
                if isinstance(attr_value, (GradLayer, Operation)):
                    self._set_mode_recursive(attr_value, mode)

    def save(self, filepath):
        params_dict = {}
        for i, param in enumerate(self.parameters()):
            params_dict[f"param_{i}"] = cp.asnumpy(param.params)  # move from GPU to CPU
        np.savez(filepath, **params_dict)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = np.load(filepath)
        for i, param in enumerate(self.parameters()):
            key = f"param_{i}"
            loaded_params = data[key]
            param.params[:] = cp.asarray(loaded_params)

    def __repr__(self):
        model_repr = "NeuralNetwork(\n"
        for layer in self.layers:
            model_repr += f"  {repr(layer)}\n"
        model_repr += ")"
        return model_repr
