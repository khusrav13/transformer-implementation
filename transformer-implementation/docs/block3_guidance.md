# Understanding Block 3 in the Transformer Layers

The transformer stacks in this repository organize each decoder layer into three conceptual blocks:

1. **Block 1 – Masked self-attention**: The decoder attends to the tokens it has already generated. This is implemented in `DecoderLayer.forward` where `self.self_attention` is called with the optional autoregressive mask and cached key/value states for fast decoding.
2. **Block 2 – Cross-attention**: The decoder attends over the encoder output (or any external memory) so it can condition generation on the source sequence. This is the `self.cross_attention` call on the normalized activations, along with the residual connection and layer normalization.
3. **Block 3 – Feed-forward transformation**: After the attention blocks, each position is processed independently by a feed-forward network. In our implementation the third block normalizes the activations, pushes them through `PositionwiseFeedForward`, applies dropout, adds the residual connection, and performs the final normalization step.

To implement or debug Block 3 you should therefore:

- Ensure the input is the output of Block 2 (after its residual connection).
- Apply the layer normalization configured for the third block before or after the feed-forward operation depending on the normalization scheme (`pre` vs. `post`).
- Run the activations through `self.feed_forward`, which is defined in `components/feedforward.py` as a position-wise multilayer perceptron with configurable activation functions (ReLU, GELU, SwiGLU, etc.).
- Add the residual connection from the block input and follow it with the appropriate normalization step so gradients remain stable.

Following these steps keeps Block 3 consistent with both the original Transformer paper and the extended features provided in this repository, such as support for SwiGLU and different normalization strategies.
