# Transformers - Detailed Explanation

This repository implements a classic Transformer model (encoder–decoder) for German → English translation, very close to the “Attention Is All You Need” architecture (Vaswani et al., 2017).  
This README walks through:

- What happens mathematically
- How each class in `model.py`, `dataset.py`, `config.py`, and `train.py` fits together
- How tensors flow (with shapes)
- Why each step exists, not just what it does

I’ll assume you already know basic PyTorch and a bit of linear algebra, but I’ll still unpack the core concepts carefully.

---

## 1. High‑level overview

### 1.1 What this project does

At a high level:

1. Loads a parallel German–English dataset (`opus_books`).
2. Builds tokenizers for source (German) and target (English).
3. Creates `BilingualDataset` that:
   - Tokenizes text
   - Adds `[SOS]`, `[EOS]`, `[PAD]`
   - Creates masks for encoder and decoder
4. Builds a full Transformer model with:
   - Input embeddings
   - Positional encodings
   - Stacked encoder and decoder blocks
   - Multi-head attention
   - Feed-forward networks
   - Residual connections and layer normalization
5. Trains the model to predict the next target token (teacher forcing).

---

### 1.2 Conceptual architecture

Think of the model like this:

```text
                +------------------------+
                |      Transformer       |
                |  (Encoder–Decoder MT)  |
                +------------------------+
                     ^              |
                     |              v
   German sentence -> Encoder   Decoder -> English sentence
```

`Encoder`: reads the **source sequence**, builds internal representation.  
`Decoder`: reads already generated (or gold) **target prefix** and attends to encoder outputs to predict the **next token**.

A more detailed ASCII diagram aligning with your code:

```text
Input (src tokens)          Target (tgt tokens)
        |                           |
        v                           v
  src_embed + src_pos        tgt_embed + tgt_pos
        |                           |
        v                           v
+---------------+             +-----------------+
|    Encoder    |             |     Decoder     |
|  N layers of  |             |   N layers of   |
|  [Self-Attn + |             | [Self-Attn +    |
|   FFN]        |             |  Cross-Attn +   |
+---------------+             |    FFN]         |
        |                     +-----------------+
        |                            |
        v                            v
   encoder_output               decoder_output
                                     |
                                     v
                          Projection (Linear + log_softmax)
                                     |
                                     v
                              Token probabilities
```

---

## 2. Data pipeline and dataset

### 2.1 Configuration (`config.py`)

`get_config()` returns a simple dictionary:

- **`batch_size`**: 8
- **`num_epochs`**: 30
- **`lr`**: \(10^{-4}\)
- **`seq_len`**: 500 (max sequence length)
- **`d_model`**: 512 (embedding/model dimension)
- **`lang_src`**: `"de"`
- **`lang_tgt`**: `"en"`
- **`model_folder`**: where weights are saved
- **`tokenizer_file`**: template for tokenizer filenames

There is also `get_weights_file_path(config, epoch)`, but note:

```python
model_basename = config['model_basename']
model_filename = f"{model_basename}{epoch}.pt"
```

Your config uses `"model_filename"` instead of `"model_basename"` – this will throw a `KeyError`.  
(Just a note; this README focuses on understanding, not fixing.)

---

### 2.2 Dataset building (`train.py` – `get_ds`)

```python
ds_raw = load_dataset('opus_books',
                      f'{config["lang_src"]}-{config["lang_tgt"]}',
                      split='train')
```

This loads a parallel dataset with structure like:

```python
item = {
    "translation": {
        "de": "German sentence here",
        "en": "English sentence here"
    }
}
```

Then:

- Builds or loads tokenizers for both languages via `get_or_build_tokenizer`.
- Splits the dataset into train and validation (90/10).
- Wraps them in `BilingualDataset`.

---

### 2.3 Tokenizers (`train.py` – `get_or_build_tokenizer`)

```python
tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
    min_frequency=2
)
tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
```

So the tokenizer:

- Splits by whitespace.
- Builds a word-level vocabulary.
- Ensures these special tokens are in the vocab:
  - `[UNK]` – unknown
  - `[PAD]` – padding
  - `[SOS]` – start of sequence
  - `[EOS]` – end of sequence

---

### 2.4 BilingualDataset (`dataset.py`)

#### 2.4.1 Initial setup

```python
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        ...
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
```

- `seq_len` is the fixed length for all sequences.
- `sos_token`, `eos_token`, `pad_token` are scalars wrapped in 1‑D tensors (`shape = (1,)`).

#### 2.4.2 `__getitem__` logic

Given an index, you:

1. Extract raw strings:

   ```python
   src_text = src_target_pair['translation'][self.src_lang]
   tgt_text = src_target_pair['translation'][self.tgt_lang]
   ```

2. Tokenize:

   ```python
   enc_input_tokens = self.tokenizer_src.encode(src_text).ids
   dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
   ```

   These are Python lists of token IDs.

3. Compute padding:

   ```python
   enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
   dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
   ```

  Why `-2` for encoder?

You add `[SOS]` and `[EOS]`, so:

$$
\text{enc\_len\_total}
= 1 + \text{len(enc\_input\_tokens)} + 1 + \text{enc\_num\_padding\_tokens}
= \text{seq\_len}
$$

For decoder input, you add only `[SOS]`, so:

$$
\text{dec\_len\_total}
= 1 + \text{len(dec\_input\_tokens)} + \text{dec\_num\_padding\_tokens}
= \text{seq\_len}
$$
4. Build `encoder_input`:

   ```python
   encoder_input = torch.cat([
       self.sos_token,
       torch.tensor(enc_input_tokens, dtype=torch.int64),
       self.eos_token,
       torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
   ])
   ```

   Shape: `(seq_len,)`

5. Build `decoder_input`:

   ```python
   decoder_input = torch.cat([
       self.sos_token,
       torch.tensor(dec_input_tokens, dtype=torch.int64),
       torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
   ])
   ```

   Shape: `(seq_len,)`

6. Build `label` (what the model should predict):

   ```python
   label = torch.cat([
       torch.tensor(dec_input_tokens, dtype=torch.int64),
       self.eos_token,
       torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
   ])
   ```

   This is shifted relative to `decoder_input`:

   ```text
   decoder_input: [SOS]  t1   t2   ...   tN   PAD PAD ...
   label:               t1   t2  ...   tN   EOS PAD PAD ...
   ```

   The model thus learns to predict the **next token** at each position.

7. Create masks:

   ```python
   "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
   ```

   Shapes:

   - `encoder_input != pad` → `(seq_len,)`
   - After `unsqueeze(0).unsqueeze(0)` → `(1, 1, seq_len)`
   - This broadcast to `(batch, 1, 1, seq_len)` during attention.

   Decoder mask:

   ```python
   "decoder_mask": (decoder_input != self.pad_token)
        .unsqueeze(0).unsqueeze(0).int()
        & casual_mask(decoder_input.size(0))
   ```

   - Padding mask: `(1, 1, seq_len)`
   - `casual_mask(size)` → `(1, seq_len, seq_len)` where positions can only attend to themselves and past tokens.
   - Bitwise `&` combines them into a mask that:
     - Hides future tokens.
     - Hides padding tokens.

#### 2.4.3 Causal mask (`casual_mask`)

```python
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
```

`torch.triu` with `diagonal=1` creates an upper triangular matrix of ones above the main diagonal:

```text
size = 4

[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

Then `mask == 0` flips it:

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

So token at position `i` can only attend to positions `0..i`.

---

## 3. Model architecture (`model.py`)

### 3.1 Input embeddings (`inputEmbeddings`)

```python
class inputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

- Input: `x` with shape `(batch, seq_len)`, containing token IDs.
- `nn.Embedding(vocab_size, d_model)` returns `(batch, seq_len, d_model)`.
- Multiplying by $\sqrt{d_{model}}$ stabilizes the scale before adding positional encodings (from the original paper).

Mathematically:

$$
\text{embed}(x_i)
= \sqrt{d_{model}} \cdot E_{x_i}
$$

where $E$ is the embedding matrix.


---

### 3.2 Positional encoding (`positionalEncoding`)

Transformers are permutation-invariant by default. Positional encoding injects information about **position**.

#### 3.2.1 Construction

```python
pe = torch.zeros(seq_len, d_model)
position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
```

For each position \(pos\) and dimension \(i\):

$$
\text{PE}_{(pos, 2i)}
= \sin\!\left(\frac{pos}{10000^{2i / d_{model}}}\right)
$$

$$
\text{PE}_{(pos, 2i+1)}
= \cos\!\left(\frac{pos}{10000^{2i / d_{model}}}\right)
$$


In code:

```python
pe[:, 0::2] = torch.sin(position * div_term)  # even indices
pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
self.register_buffer('pe', pe)
```

- `register_buffer` means `pe` is saved with the model but not a trainable parameter.

#### 3.2.2 Forward

```python
def forward(self, x):
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)
```

- `x` shape: `(batch, seq_len, d_model)`
- `self.pe[:, :seq_len, :]` shape: `(1, seq_len, d_model)`
- Broadcast to `(batch, seq_len, d_model)`.

ASCII intuition:

```text
Token embeddings:     [word1_vec, word2_vec, ..., wordN_vec]
Positional encodings: [pos1_vec,  pos2_vec,  ..., posN_vec ]
                         |            |
Add elementwise:      word1_vec + pos1_vec, ...
```

---

### 3.3 Layer normalization (`LayerNormalization`)

Layer normalization stabilizes training.

Mathematically, for each position vector $x \in \mathbb{R}^{d}$:

$$
\begin{aligned}
\mu &= \frac{1}{d} \sum_{i=1}^{d} x_i \\
\sigma &= \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}
\end{aligned}
$$

$$
\text{LN}(x)
= \alpha \cdot \frac{x - \mu}{\sigma + \epsilon}
+ \beta
$$

In code:

```python
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
return self.alpha * (x - mean) / (std + self.eps) + self.bias
```

- `dim=-1` means normalization over the feature dimension \(d_{model}\).

---

### 3.4 Feed-forward network (`FeedForwardNetwork`)

This is the same in each encoder and decoder layer.

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

Shapes:

- Input: `(batch, seq_len, d_model)`
- `linear_1`: `(batch, seq_len, d_ff)`
- ReLU + dropout
- `linear_2`: `(batch, seq_len, d_model)`

Mathematically:

$$
\text{FFN}(x)
= W_2 \cdot \text{Dropout}\!\left(\text{ReLU}(W_1 x + b_1)\right)
+ b_2
$$


This gives each token position its own small MLP, applied independently to each position.

---

### 3.5 Multi-head attention (`MultiHeadAttention`)

Attention lets each token look at other tokens and decide how much to “pay attention” to them.

#### 3.5.1 Parameterization

```python
self.w_q = nn.Linear(d_model, d_model)
self.w_k = nn.Linear(d_model, d_model)
self.w_v = nn.Linear(d_model, d_model)
self.w_o = nn.Linear(d_model, d_model)
self.h = h         # number of heads
self.d_k = d_model // h
```

Given:

- Query $Q$, Key $K$, Value $V$ — all of shape `(batch, seq_len, d_model)`.

They get projected:

$$
Q' = Q W_Q,
\quad
K' = K W_K,
\quad
V' = V W_V
$$


These are reshaped into multiple heads.

#### 3.5.2 Reshaping into heads

```python
query = query.view(batch, seq_len, h, d_k).transpose(1, 2)
# shape: (batch, h, seq_len, d_k)
```

Same for `key`, `value`.

ASCII shape transformation:

```text
Before: (batch, seq_len, d_model)
After splitting:
        (batch, seq_len, h, d_k)
Transpose:
        (batch, h, seq_len, d_k)
```

Each head performs attention independently.

#### 3.5.3 Scaled dot-product attention

Static method:

```python
attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
if mask is not None:
    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
attention_scores = attention_scores.softmax(dim=-1)
...
return (attention_scores @ value), attention_scores
```

Mathematically:

1. **Compute raw similarity:**

   $$
   \text{scores}
   = \frac{Q K^\top}{\sqrt{d_k}}
   $$

2. **Apply mask:**

   - Positions where `mask == 0` become $-10^9$, so after softmax the probability is $\approx 0$.

3. **Softmax:**

   $$
   \alpha
   = \text{softmax}(\text{scores})
   $$

4. **Weighted sum:**

   $$
   \text{Attention}(Q, K, V)
   = \alpha V
   $$


#### 3.5.4 Combining heads

Result of attention per head: `(batch, h, seq_len, d_k)`:

```python
x = x.transpose(1, 2).contiguous().view(batch, seq_len, h * d_k)
return self.w_o(x)
```

- After transpose: `(batch, seq_len, h, d_k)`
- Then flatten: `(batch, seq_len, d_model)`
- Finally apply output linear `w_o`.

---

### 3.6 Residual connections (`ResidualConnection`)

```python
def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
```

Pattern:

1. Normalize input: `norm(x)`
2. Apply sublayer (attention or feed-forward).
3. Apply dropout.
4. Add original `x` (residual connection).

This helps gradients flow and stabilizes training.

ASCII:

```text
x ----> LN ----> Sublayer ----> Dropout ----+
 |                                          |
 +------------------------------------------+
```

---

### 3.7 Encoder (`EncoderBlock` and `Encoder`)

#### 3.7.1 Single encoder block

```python
class EncoderBlock(nn.Module):
    def __init__(..., self_attention_block, feed_forward_block, dropout):
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x,
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
```

Structure:

1. Self‑attention with residual & layer norm.
2. Feed‑forward with residual & layer norm.

ASCII:

```text
Input x
  |
  |--[LN]-->[Self-Attn with mask]-->[Dropout]--+
  |                                            |
  +--------------------------------------------+  -> x1
  |
  |--[LN]-->[FFN]-->[Dropout]------------------+
  |                                            |
  +--------------------------------------------+  -> output
```

#### 3.7.2 Stacked encoder

```python
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

This stacks `N` encoder blocks, then applies final layer normalization.

---

### 3.8 Decoder (`DecoderBlock` and `Decoder`)

#### 3.8.1 Single decoder block

```python
class DecoderBlock(nn.Module):
    def __init__(..., self_attention_block, cross_attention_block, feed_forward_block, dropout):
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
```

Clearly:
1. **Masked self‑attention** over target sequence (uses `tgt_mask`).
2. **Cross‑attention**: Query = decoder states, Key/Value = encoder output.
3. **Feed‑forward** network.

ASCII:

```text
Decoder input x
  |
  |--[LN]-->[Masked Self-Attn (tgt_mask)]-->[Dropout]--+
  |                                                   |
  +---------------------------------------------------+  -> x1
  |
  |--[LN]-->[Cross-Attn (Q=x1, K=V=encoder_output)]-->[Dropout]--+
  |                                                              |
  +--------------------------------------------------------------+ -> x2
  |
  |--[LN]-->[FFN]-->[Dropout]------------------------------------+
  |                                                              |
  +--------------------------------------------------------------+ -> output
```

#### 3.8.2 Stacked decoder

```python
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
```

Same pattern as encoder: stack `N` blocks, then final `LayerNormalization`.

---

### 3.9 Projection to vocabulary (`ProjectionLayer`)

```python
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        return torch.log_softmax(self.proj(x), dim=-1)
```

- Input: `(batch, seq_len, d_model)`
- Output: `(batch, seq_len, vocab_size)` – log probabilities over tokens.
$$
\text{logits}_{t}
= x_{t} W + b,
\quad
\text{output}_{t}
= \log \text{softmax}(\text{logits}_{t})
$$

This is what the loss is computed on.

---

### 3.10 Full Transformer wrapper (`Transformers` class)

```python
class Transformers(nn.Module):
    def __init__(self, encoder, decoder,
                 src_embed, tgt_embed,
                 src_pos, tgt_pos,
                 projection_layer):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
```

Encapsulates all pieces. Methods:

```python
def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.tgt_embed(tgt)
    tgt = self.tgt_pos(tgt)
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

def project(self, x):
    return self.projection_layer(x)
```

So training uses:

```text
encoder_output = model.encode(encoder_input, encoder_mask)
decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
log_probs      = model.project(decoder_output)
```

---

### 3.11 Model building function (`build_transform`)

```python
def build_transform(src_vocab_size, tgt_vocab_size,
                    src_seq_len, tgt_seq_len,
                    d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
```

Steps:

1. **Embeddings & positions**:
   ```python
   src_embed = inputEmbeddings(d_model, src_vocab_size)
   tgt_embed = inputEmbeddings(d_model, tgt_vocab_size)
   src_pos   = positionalEncoding(d_model, src_seq_len, dropout)
   tgt_pos   = positionalEncoding(d_model, tgt_seq_len, dropout)
   ```

2. **Encoder blocks**:
   ```python
   encoder_blocks = []
   for _ in range(N):
       encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
       feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
       encoder_block = EncoderBlock(d_model, encoder_self_attention_block,
                                    feed_forward_block, dropout)
       encoder_blocks.append(encoder_block)
   encoder = Encoder(nn.ModuleList(encoder_blocks))
   ```

3. **Decoder blocks** similarly (with self and cross attention).

4. **Projection layer**:
   ```python
   projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
   ```

5. **Combine into Transformer**:
   ```python
   transformer = Transformers(encoder, decoder, src_embed, tgt_embed,
                              src_pos, tgt_pos, projection_layer)
   ```

6. **Parameter initialization**:
   ```python
   for p in transformer.parameters():
       if p.dim() > 1:
           nn.init.xavier_uniform_(p)
   ```

Xavier uniform is used for linear layers; it keeps variance fairly stable from layer to layer.

---

## 4. Training loop (`train.py`)

### 4.1 Getting model

```python
def get_model(config, vocab_src_len, vocal_tgt_len):
    model = build_transform(
        vocab_src_len,
        vocal_tgt_len,
        config['seq_len'],
        config['seq_len'],
        config['d_model']
    )
    return model
```

Note the small typo `vocal_tgt_len` in parameter name – logical value is target vocab size.

---

### 4.2 Training setup (`train_model`)

1. **Device**:

   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Data**:

   ```python
   train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
   ```

3. **Model to device**:

   ```python
   model = get_model(config, tokenizer_src.get_vocab_size(),
                              tokenizer_tgt.get_vocab_size()).to(device)
   ```

4. **Optimizer**:

   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
   ```

5. **Loss function**:

   ```python
   loss_fn = nn.CrossEntropyLoss(
       ignore_index=tokenizer_src.token_to_id('[PAD]'),
       label_smoothing=0.1
   ).to(device)
   ```

   - `ignore_index`: padding tokens are ignored in the loss.
   - `label_smoothing=0.1`: targets are smoothed a bit to prevent overconfidence.

   For each position, instead of one-hot target of 1 on correct class, 0 elsewhere, it uses:

$$
y_{\text{smooth}}
= (1 - \epsilon) \cdot y_{\text{one-hot}}
+ \frac{\epsilon}{V} \cdot \mathbf{1}
$$

where $\epsilon = 0.1$ and $V$ is the vocabulary size.

---

### 4.3 Core training loop

```python
for epoch in range(initial_epoch, config['num_epochs']):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
    for batch in batch_iterator:
        encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
        decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)

        encoder_mask = batch['encoder_mask'].to(device)    # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device)    # (B, 1, seq_len, seq_len)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask,
                                      decoder_input, decoder_mask)

        proj_op = model.project(decoder_output)            # (B, seq_len, vocab_tgt)
        label = batch['label'].to(device)                  # (B, seq_len)

        loss = loss_fn(
            proj_op.view(-1, tokenizer_tgt.get_vocab_size()),
            label.view(-1)
        )
```

Shapes during loss computation:

- `proj_op.view(-1, vocab_size)` → `(B * seq_len, vocab_size)`
- `label.view(-1)` → `(B * seq_len,)`

Then:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

### 4.4 Intuition of what the model learns

At each decoder time step \(t\):

1. The decoder sees:
   - All encoder outputs (full source sentence).
   - Target tokens \(y_{<t}\) (due to masking).

2. It produces a distribution \(p(y_t | y_{<t}, x)\) over the vocabulary.

Training teaches the model to maximize likelihood:

$$
\prod_{t} p(y_t \mid y_{<t}, x)
$$

Which is equivalent to minimizing cross-entropy loss between predicted distribution and true next token.

---

## 5. Tensor shape flow summary

Here’s a compact view of shapes through the model for a batch:

```text
Batch size: B
Sequence length: S = seq_len
Model dimension: D = d_model
Heads: H
Per-head dim: d_k = D / H
Source vocab: V_src
Target vocab: V_tgt
```

### 5.1 Encoder path

```text
encoder_input:        (B, S)
src_embed:            (B, S, D)
+ src_pos:            (B, S, D)
encoder_mask:         (B, 1, 1, S)

Inside each layer:
  Self-attn Q,K,V:    (B, S, D)
  -> split heads:     (B, H, S, d_k)
  Scores:             (B, H, S, S)
  Softmax + mask
  Attn output:        (B, H, S, d_k)
  -> combine heads:   (B, S, D)
  + residual etc.

Final encoder_output: (B, S, D)
```

### 5.2 Decoder path

```text
decoder_input:        (B, S)
tgt_embed + tgt_pos:  (B, S, D)
decoder_mask:         (B, 1, S, S)

Inside each decoder layer:
  Masked self-attn:
    Q,K,V:            (B, S, D)
    Mask:             (B, 1, S, S)

  Cross-attn:
    Q: decoder states (B, S, D)
    K,V: encoder_out  (B, S, D)
    src_mask:         (B, 1, 1, S)

  FFN per position:   (B, S, D) -> (B, S, D)

Final decoder_output: (B, S, D)
Projection:           (B, S, V_tgt)
```

---

## 6. ASCII mental model of encoder–decoder attention

Think of the cross‑attention step as creating soft alignments between source and target tokens.

```text
Source (encoder output):

[S1]---\
[S2]----\______________
[S3]---------------\   \
[S4]----------------\___\____

Target (decoder state at t):

[T_t] -----------------------> attended representation

Weights (per head):

   S1  S2  S3  S4
Tt 0.1 0.2 0.6 0.1

So decoder position t "looks" mostly at S3.
```

Multi‑head attention repeats this with different learned projections, so different heads can focus on different patterns (e.g., syntax, long‑range dependencies, etc.).

---

