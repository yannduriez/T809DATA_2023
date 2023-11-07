import numpy as np
from tqdm import tqdm

from encoder import get_encoder
from tools import get_params


# Multi-head attention

def softmax(x):
    num = np.exp(x - np.max(x, axis=1, keepdims=True))
    den = np.sum(num, axis=1, keepdims=True)
    
    return num / den


def attention(Q, K, V):
    product = (Q @ K.T) / np.sqrt(Q.shape[1])
    
    return softmax(product) @ V


def masked_attention(Q, K, V, mask):
    product = (Q @ K.T) / np.sqrt(Q.shape[1]) + mask
    
    return softmax(product) @ V


def linear_projection(x, w, b):
    
    return x @ w + b


def multi_head_attention(x, attn, number_of_heads):
    w_1, b_1 = attn["c_attn"]["w"], attn["c_attn"]["b"]
    w_2, b_2 = attn["c_proj"]["w"], attn["c_proj"]["b"]
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    
    projected_x = linear_projection(x, w_1, b_1)
    Q, K, V = np.split(projected_x, 3, axis=1)

    Q = np.split(Q, number_of_heads, axis=1)
    K = np.split(K, number_of_heads, axis=1)
    V = np.split(V, number_of_heads, axis=1)
    
    output_heads_list = []
    for i in range(number_of_heads):
        output_head = masked_attention(Q[i], K[i], V[i], mask)
        output_heads_list.append(output_head)
    
    merged_heads = np.concatenate(output_heads_list, axis=1)
    x = linear_projection(merged_heads, w_2, b_2)

    return x


# Transformer blocks and GPT2


def gelu(x):
    g = 0.5 * x *(1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

    return g

def layer_normalization(x, g, b, eps=1e-5):
    mu = np.mean(x, axis=1, keepdims=True)
    sigma = np.var(x, axis=1, keepdims=True)
    
    normal = (x - mu) / np.sqrt(sigma + eps)

    return g * normal + b


def feed_forward_network(x, mlp):
    w_1, b_1 = mlp["c_fc"]["w"], mlp["c_fc"]["b"]
    w_2, b_2 = mlp["c_proj"]["w"], mlp["c_proj"]["b"]
    
    projected_x = linear_projection(x, w_1, b_1)

    g = gelu(projected_x)

    return linear_projection(g, w_2, b_2)


def transformer_block(x, block, number_of_heads):
    mlp, attn = block["mlp"], block["attn"]
    ln_1, ln_2 = block["ln_1"], block["ln_2"]
    g_1, b_1, g_2, b_2 = ln_1["g"], ln_1["b"], ln_2["g"], ln_2["b"]
    
    normalized_layer_1 = layer_normalization(x, g_1, b_1)
    forward_pass = multi_head_attention(normalized_layer_1, attn, number_of_heads)
    input_x_added = forward_pass + x
    x_store = input_x_added

    normalized_layer_2 = layer_normalization(input_x_added, g_2, b_2)
    feed_forward_out = feed_forward_network(normalized_layer_2, mlp)
    stored_x_added = feed_forward_out + x_store
    
    return stored_x_added


def gpt2(inputs, wte, wpe, blocks, ln_f, number_of_heads):
    g_final, b_final = ln_f["g"], ln_f["b"]
    x = wte[inputs] + wpe[range(len(inputs))]  # Step 1: Sum positional encoding and token encoding 
    
    transformed_block = transformer_block(x, blocks, number_of_heads)
    normalized_layer = layer_normalization(transformed_block, g_final, b_final)

    return normalized_layer @ wte.T


def generate(input_text, tokens_to_generate=40, model_size="124M", models_dir="models", loading_bar=True):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    
    hparams, params = get_params(model_size, models_dir)
    encoder = get_encoder(model_size, models_dir)
    number_of_heads = hparams["n_head"]
    max_context = hparams["n_ctx"]

    # Port the input text to ids
    input_ids = encoder.encode(input_text)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + tokens_to_generate < max_context

    # generate output ids
    output_ids = []
    
    if loading_bar:
        loop_range = tqdm(range(tokens_to_generate), "Thinking...")
    else:
        loop_range = range(tokens_to_generate)

    for _ in loop_range:
        # Call our gtp2 model with input plus generated tokens
        output = gpt2(input_ids + output_ids, **params, number_of_heads=number_of_heads) 

        # Get the next token from the output
        next_id = np.argmax(output[-1])

        # Save the result
        output_ids.append(int(next_id))

    # Port the output ids to text
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    
    Test your implemetntation with something like this:
    print(generate("Hello! How do you do?"))

    You can try out different sized models from this list: ["124M", "355M", "774M", "1558M"]
    Make sure you have enough space on your device since the bigger models are quite large.
    """
    
    '''
    #1.1
    #print(softmax(np.array([[-1., 0.], [0.2, 1.]])))

    #1.2
    np.random.seed(4321)
    q = np.random.rand(3,2)
    k = np.random.rand(3,2)
    v = np.random.rand(3,2)
    x = attention(q, k, v)
    print(x)
    
    #1.3
    np.random.seed(4321)
    nf = 10
    q = np.random.rand(nf,2)
    k = np.random.rand(nf,2)
    v = np.random.rand(nf,2)
    mask = (1 - np.tri(nf)) * -1e10
    x = masked_attention(q, k, v, mask)
    print(x)
    
    ## Section 2
    #2.1
    np.random.seed(4321)
    x = np.random.rand(3,2)
    w = np.random.rand(2,3)
    b = np.random.rand(3,1)
    lp = linear_projection(x, w, b)
    print(lp)
    
    #2.2
    np.random.seed(4321)
    x = np.random.rand(3,4)
    w_1 = np.random.rand(4,12)
    b_1 = np.random.rand(3,1)
    w_2 = np.random.rand(4,3)
    b_2 = np.random.rand(3,1)
    attn = {"c_attn": {"w": w_1, "b": b_1}, "c_proj": {"w": w_2, "b": b_2}}
    x = multi_head_attention(x, attn, 2)
    print(x)
    '''

    # Transformer blocks and GPT2
    '''
    #1.1
    print(gelu(np.array([[-1., 0.], [0.2,  1.]])))
    
    #1.2
    np.random.seed(4321)
    x = np.random.rand(3,2)
    g = np.random.rand(3,2)
    b = np.random.rand(3,1)
    ln = layer_normalization(x, g, b)
    print(ln)
    
    #2.1
    np.random.seed(4321)
    x = np.random.rand(3,4)
    w_1 = np.random.rand(4,5)
    b_1 = np.random.rand(3,1)
    w_2 = np.random.rand(5,4)
    b_2 = np.random.rand(3,1)
    mlp = {"c_fc": {"w": w_1, "b": b_1}, "c_proj": {"w": w_2, "b": b_2}}
    x = feed_forward_network(x, mlp)
    print(x)
    '''

    #2.4
    print(generate("Hello! How are you?"))