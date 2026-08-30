# SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

## Appendix Variable Names (Notation)
SparseVLM uses a set of variable names, consistent with the original paper, to describe the pruning and selection process:

- **SB**: The raw attention score matrix between text queries (Q) and visual keys (K), computed as $SB = QK^T/\sqrt{d_k}$. This matrix quantifies how much each text token attends to each visual token before normalization.
- **PB**: The attention probability matrix, obtained by applying a row-wise softmax to SB ($PB = \text{softmax}(SB)$). Each row sums to 1, representing the distribution of attention from a text token over all visual tokens.
- **V**: The special vector ("raters") that selects which text tokens guide the pruning. $V$ is a vector with $1/n$ at rater positions (selected text tokens) and $0$ elsewhere, so that $V^T PB$ averages the attention of the chosen raters.
- **OB**: The mean attention score for each visual token, computed as $OB = V^T PB$ (sometimes denoted as Ov). This gives a single importance score per visual token, reflecting its relevance to the selected text raters.
- **O / Ov**: The importance scores for visual tokens, typically $Ov = OB$; these scores are used to decide which visual tokens to keep or prune.
- **Ok**: The top-k importance values selected from Ov, representing the most relevant visual tokens for the current image block.
- **Ik**: The indices of the top-k visual tokens to keep within each image block, used to construct the mask for pruning or recycling.

## What is SparseVLM?
SparseVLM is a method for efficient inference in vision-language models (VLMs) that introduces text-guided visual token pruning. The core idea is to use the model’s own attention mechanisms—specifically, the cross-attention between text and visual tokens—to identify and retain only the most relevant visual tokens (image patches) for a given text prompt. By doing so, SparseVLM reduces computational cost and memory usage, all without retraining or fine-tuning the model. It is a training-free, plug-and-play approach that leverages the internal attention structure of transformer-based VLMs to dynamically score and select visual tokens for each input.

## What Does SparseVLM Do?
SparseVLM operates in two main steps:
1. **Text-Guided Pruning:** The model computes attention scores (SB) between text tokens (queries) and visual tokens (keys) using its cross-attention layers. By analyzing these scores, it identifies a subset of visual tokens that are most relevant to the text, and prunes the rest. The selection of which text tokens act as "raters" (V) can be based on their mean attention or tailored to the question context.
2. **Token Recycling:** Instead of removing pruned tokens, SparseVLM can compress them (e.g., by averaging with kept tokens) to maintain the original sequence length, which is often required by certain models. This ensures compatibility with models like Gemma 3 that expect a fixed number of tokens, and allows the model to operate without architectural changes. Pruned tokens are merged with kept tokens (e.g., by averaging their embeddings or image patches), so all original tokens are preserved in a compressed form. This step is crucial for models with tightly integrated architectures, as it avoids disrupting the expected input structure.

This approach enables significant speedups and memory savings during inference, especially for large images or multi-image inputs, while preserving answer quality. It also provides interpretability, as the attention and pruning decisions can be visualized and analyzed.

## Multi-Image Implementation vs. Single Image (Original Paper)
The original SparseVLM paper primarily addresses single-image scenarios, where the model processes one image at a time and prunes visual tokens based on their relevance to the text. In this work, SparseVLM is extended to handle **multi-image settings**, such as multimodal table question answering where each table cell may contain a different image. This adaptation required several key changes:

**Key differences in the multi-image implementation:**
- **Per-Image Block Processing:** Each image’s visual tokens are treated as a separate block. Attention scores, pruning, and normalization are computed independently for each image block, rather than aggregating across all images. This ensures that each image is evaluated on its own merits, without interference from the sequential order of images in the input.
- **Independent Normalization:** The importance scores (Ov) for each image are normalized separately. In early experiments, global normalization led to a strong positional bias: initial images were pruned too aggressively, while later images retained too many tokens. By normalizing per image, the pruning is fair and interpretable for all images.
- **Token Recycling:** When pruning, pruned tokens are compressed (e.g., by averaging with nearest kept tokens) rather than removed, ensuring compatibility with models that require fixed-length inputs. This is especially important for architectures like Gemma 3, which expect a fixed number of tokens and tightly integrate visual and text processing.

## LLaVA vs. Gemma 3: Architectural Differences

- **LLaVA:** Uses a modular architecture with a separate vision encoder (e.g., CLIP) and a lightweight connector to the language model. Visual features are extracted independently and then injected into the language model. This separation allows for flexible integration but can limit the depth of interaction between modalities.
- **Gemma 3:** Features a tightly integrated, unified architecture where visual information is processed directly within the main language transformer. Visual tokens are embedded and handled alongside text tokens, allowing for deeper fusion and more flexible attention patterns. This design requires that all tokens (text and visual) be present in a fixed-length sequence, making token recycling essential for compatibility.

**Why are these changes necessary?**
- In Gemma 3’s unified architecture, all tokens (text and visual) are processed together, and the model expects a fixed sequence length. This makes token recycling (rather than removal) essential to avoid breaking the model’s input expectations.
- The sequential processing of images in a single sequence can cause attention normalization issues—if all images share a global normalization, the model may focus disproportionately on later images. Independent per-image normalization ensures fair and interpretable pruning, preventing positional bias.

**Drawbacks:**
- **Loss of Global Context:** By normalizing and pruning each image independently, the model may miss cross-image relationships that could be important for some tasks, such as reasoning across multiple images in a table.
- **Implementation Complexity:** Handling token recycling and per-image normalization adds complexity compared to the original single-image, global-pruning approach. The codebase must carefully track image blocks, masks, and ensure that all tokens are handled correctly.

## Summary of Changes from the Original Approach

Throughout development, several changes and improvements were made to adapt SparseVLM for multimodal table QA and multi-image settings:

1. **Robust Image Handling:** Error handling was added for missing files and non-RGB images, ensuring that dummy images are used as fallbacks and all images are in the correct format.
2. **Softmax Normalization:** The Ov importance scores are now normalized independently for each image block, preventing positional bias and ensuring fair pruning.
3. **Independent Block Processing:** Each image block is processed independently, with its own attention, pruning, and normalization, rather than aggregating across all images.
4. **Token Recycling:** Instead of removing pruned tokens, they are compressed (e.g., by averaging with kept tokens) to maintain the original sequence length, as required by Gemma 3.


## Reflections: Why SparseVLM Struggles for Multimodal Tables

While SparseVLM provides an efficient and interpretable approach to visual token pruning, it faces significant challenges in the context of multimodal tables:

- **Lack of Cross-Image Reasoning:** By treating each image independently, the model cannot reason about relationships or dependencies between images in different table cells. Many table QA tasks require comparing or aggregating information across multiple images.
- **Positional and Sequential Bias:** Even with independent normalization, the model’s underlying architecture may still introduce subtle biases based on the order of images in the input sequence.
- **Complexity of Table Structure:** Multimodal tables often contain a mix of text and images, with complex spatial and semantic relationships. SparseVLM’s per-image pruning does not account for these higher-level structures.
- **Loss of Cross-Image Context:** Pruning and recycling tokens independently for each image can cause the model to miss patterns or context that span multiple images or the entire table.
- **Multihop Reasoning Limitation:** If answering a query requires reasoning through intermediate images (not directly referenced in the query), this method may prune important parts of those images, making multihop or indirect reasoning difficult.

In summary, while SparseVLM is effective for single-image or simple multi-image tasks, its current form is not well-suited for the complex, interconnected reasoning required by multimodal tables. Future work may require new methods that explicitly model cross-image and cross-modal relationships within tables.