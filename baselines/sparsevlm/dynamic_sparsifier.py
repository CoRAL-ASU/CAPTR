"""
Dynamic Token Sparsifier for SparseVLM with Gemma Models
Implements multi-layer progressive pruning using forward hooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

PRUNING_LAYERS = [3, 9, 18]  # Multi-layer progressive pruning (adapted from LLaVA's [2, 6, 15])
TOKEN_BUDGETS = {
    192: [300, 200, 118],  # tokens to keep at each layer
    128: [238, 108, 60],
    96: [246, 54, 28],
    64: [66, 34, 20]
}

class DynamicTokenSparsifier:
    """
    Dynamic multi-layer token sparsification using forward hooks.
    Implements SparseVLM's progressive pruning at multiple decoder layers.
    Uses token averaging instead of removal to maintain sequence length (required for Gemma).
    """
    
    def __init__(self, model, processor, pruning_layers=None, topk_ratio=0.7):
        self.model = model
        self.processor = processor
        self.pruning_layers = pruning_layers if pruning_layers is not None else PRUNING_LAYERS
        self.topk_ratio = topk_ratio
        self.hooks = []
        self.layer_data = {} 
        self.visual_token_info = None
        self.text_rater_indices = None
        
    def set_visual_token_info(self, blocks, question_token_indices):
        """Set information about visual tokens and text raters before forward pass."""
        self.visual_token_info = blocks
        self.text_rater_indices = question_token_indices
    
    def set_visual_token_info_batch(self, all_blocks, all_question_indices, batch_size):
        """Set information for batched processing.
        
        For now, uses the first example's info as representative.
        Future: Could merge blocks/indices across batch for better sparsification.
        """
        if batch_size > 0:
            self.visual_token_info = all_blocks[0]
            self.text_rater_indices = all_question_indices[0]
        else:
            self.visual_token_info = None
            self.text_rater_indices = None
        
    def register_hooks(self):
        """Register forward PRE-hooks at specified pruning layers.
        
        Uses PRE-hooks instead of post-hooks to modify hidden states before the layer's attention computation
        1. Token merging happens before attention
        2. Merged tokens are present during attention
        3. Subsequent layers see the merged representations
        """
        self.remove_hooks()  # Clear any existing hooks
        
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
            layers = self.model.model.language_model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
            layers = self.model.language_model.model.layers
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'layers'):
            layers = self.model.language_model.layers
        
        if layers is None:
            raise AttributeError(f"Cannot find model layers. Model type: {type(self.model)}")
            
        for layer_idx in self.pruning_layers:
            if layer_idx < len(layers):
                pre_hook = layers[layer_idx].register_forward_pre_hook(
                    self._make_pre_sparsification_hook(layer_idx)
                )
                self.hooks.append(pre_hook)
                
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_data = {}
        
    def _make_pre_sparsification_hook(self, layer_idx):
        """Create a PRE-forward hook that modifies input BEFORE layer computation.
        """
        def pre_hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                
                # Apply sparsification
                if self.visual_token_info and self.text_rater_indices:
                    # Modify hidden states BEFORE layer processes them
                    hidden_states = self._apply_sparsification(
                        hidden_states, layer_idx
                    )
                    
                    # Return modified input tuple
                    return (hidden_states,) + input[1:]
            return input
        return pre_hook
        
    def _apply_sparsification(self, hidden_states, layer_idx):
        """
        Apply token sparsification using ACTUAL MERGING (not averaging).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        if seq_len == 1:
            return hidden_states
        
        for block_idx, (vstart, vend) in enumerate(self.visual_token_info):
            if vend >= seq_len:
                continue
                
            v_len = vend - vstart + 1
            
            # Extract visual and text hidden states
            vis_hidden = hidden_states[:, vstart:vend+1, :]  # (B, L_vis, D)
            
            # Use text rater indices to compute importance
            if self.text_rater_indices is not None and len(self.text_rater_indices) > 0:
                text_hidden = hidden_states[:, self.text_rater_indices, :]  # (B, L_text, D)
                
                # Compute attention scores between text and visual tokens
                # Normalize for stability
                vis_norm = F.normalize(vis_hidden, dim=-1)
                text_norm = F.normalize(text_hidden, dim=-1)
                
                # Attention: (B, L_text, D) @ (B, D, L_vis) -> (B, L_text, L_vis)
                attention_scores = torch.bmm(text_norm, vis_norm.transpose(1, 2))
                
                # Average across text raters to get importance per visual token
                importance = attention_scores.mean(dim=1)  # (B, L_vis)
            else:
                importance = vis_hidden.norm(dim=-1)  # (B, L_vis)
                
            # Determine how many tokens to keep at this layer
            layer_position = self.pruning_layers.index(layer_idx)
            
            # Dynamic budget based on actual visual tokens and topk_ratio
            target_keep_ratio = self.topk_ratio * (1.0 - 0.15 * layer_position)
            keep_count = max(16, int(v_len * target_keep_ratio))
            
            # Critical: bound by actual importance tensor size
            actual_tokens = importance.shape[1]
            keep_count = min(keep_count, actual_tokens - 1)
            keep_count = max(1, keep_count)
            
            if keep_count >= actual_tokens:
                continue
            
            # Select top-k most important tokens
            _, top_indices = torch.topk(importance, k=keep_count, dim=1, largest=True)
            
            # Create mask for kept tokens
            kept_mask = torch.zeros(batch_size, actual_tokens, dtype=torch.bool, device=device)
            kept_mask.scatter_(1, top_indices, True)
            
            # Merges pruned tokens into their nearest kept tokens
            # The kept token becomes a "super-token" representing merged information
            for b in range(batch_size):
                kept_idx = top_indices[b].sort()[0]
                pruned_idx = (~kept_mask[b]).nonzero(as_tuple=True)[0]
                
                if len(pruned_idx) > 0 and len(kept_idx) > 0:
                    # Group pruned tokens by their nearest kept token
                    merge_groups = {k.item(): [] for k in kept_idx}
                    
                    for p_idx in pruned_idx:
                        # Find nearest kept token
                        distances = torch.abs(kept_idx.float() - p_idx.float())
                        nearest_kept = kept_idx[distances.argmin()].item()
                        merge_groups[nearest_kept].append(p_idx.item())
                    
                    # Merge each kept token accumulates information from its group
                    for kept_pos, pruned_positions in merge_groups.items():
                        if len(pruned_positions) > 0:
                            # Weighted merge kept token gets more weight
                            merge_weight = 1.0 / (1.0 + len(pruned_positions))
                            
                            # Starts with kept token scaled down
                            merged = hidden_states[b, vstart + kept_pos] * merge_weight
                            
                            # Add contributions from pruned tokens
                            for p_pos in pruned_positions:
                                merged += hidden_states[b, vstart + p_pos] * merge_weight
                            
                            hidden_states[b, vstart + kept_pos] = merged
                            
                            # Zero out pruned positions
                            # This signals to attention that these are inactive
                            for p_pos in pruned_positions:
                                hidden_states[b, vstart + p_pos] = 0.0
                        
        return hidden_states


def select_text_raters_exact(
    full_token_ids: torch.Tensor,
    question_token_ids: torch.Tensor,
    visual_blocks: List[Tuple[int, int]]
) -> List[int]:
    """
    Select text tokens (raters) using exact token matching.
    This is more robust than string matching.
    
    Args:
        full_token_ids: Full input token sequence
        question_token_ids: Question tokens (without special tokens)
        visual_blocks: List of (start, end) positions for visual token blocks
    
    Returns:
        List of indices corresponding to question tokens in the full sequence
    """
    device = full_token_ids.device
    question_token_ids = question_token_ids.to(device)
    
    question_token_indices = []
    q_len = len(question_token_ids)
    
    for i in range(len(full_token_ids) - q_len + 1):
        if torch.equal(full_token_ids[i:i + q_len], question_token_ids):
            question_token_indices = list(range(i, i + q_len))
            break
    
    if not question_token_indices and len(visual_blocks) > 0:
        last_visual_end = max(end for _, end in visual_blocks)
        question_token_indices = list(range(last_visual_end + 1, len(full_token_ids)))
    
    return question_token_indices
