"""
debug_explainer.py - Diagnostic précis du problème explainer
"""

import torch
import torch.nn as nn
import sys
sys.path.append("src")

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from torch_geometric.loader import DataLoader
import pickle
from captum.attr import LayerDeepLift


class DebugWrapper(nn.Module):
    """Wrapper de debug pour voir exactement ce que reçoit Captum"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.shared_transformer = model.shared_transformer
        self.classifier_coarse = model.classifier_coarse
        
    def forward(self, *args, **kwargs):
        print(f"\n🔧 DEBUG WRAPPER CALL:")
        print(f"   args count: {len(args)}")
        print(f"   kwargs: {kwargs}")
        
        for i, arg in enumerate(args):
            print(f"   arg[{i}]: type={type(arg)}, shape={getattr(arg, 'shape', 'no shape')}")
            if hasattr(arg, 'shape'):
                print(f"           min={arg.min().item():.4f}, max={arg.max().item():.4f}")
        
        # Test différentes stratégies de parsing
        if len(args) == 1:
            # Un seul argument
            single_arg = args[0]
            print(f"\n   STRATÉGIE 1: Single tensor input")
            print(f"   Shape: {single_arg.shape}")
            
            if len(single_arg.shape) == 2:
                # [total_features, 1] ou [B, total_features] ?
                print(f"   Possibilité: [batch, flattened_features]")
                B = single_arg.shape[0] 
                total_features = single_arg.shape[1]
                print(f"   B={B}, total_features={total_features}")
                
                # Essayer de retrouver x_embeddings + demographic_info
                # x_embeddings: [B, 18, 128] = B * 18 * 128 features
                # demographic_info: [B, 2] = B * 2 features  
                # Total: B * (18*128 + 2) = B * 2306
                
                expected_x_features = 18 * 128  # 2304
                expected_demo_features = 2
                expected_total = expected_x_features + expected_demo_features  # 2306
                
                print(f"   Expected total features per sample: {expected_total}")
                print(f"   Actual features per sample: {total_features}")
                
                if total_features == expected_total:
                    print("   ✅ MATCH! Captum a flatté les inputs")
                    # Reconstruction
                    x_embeddings_flat = single_arg[:, :expected_x_features]  # [B, 2304]
                    demographic_info = single_arg[:, expected_x_features:]   # [B, 2]
                    
                    x_embeddings = x_embeddings_flat.view(B, 18, 128)  # [B, 18, 128]
                    
                    print(f"   Reconstructed x_embeddings: {x_embeddings.shape}")
                    print(f"   Reconstructed demographic_info: {demographic_info.shape}")
                    
                    # Test forward
                    return self._forward_with_reconstructed(x_embeddings, demographic_info)
                else:
                    print("   ❌ Pas de match - format inconnu")
                    
        elif len(args) == 2:
            print(f"\n   STRATÉGIE 2: Two separate inputs")
            arg1, arg2 = args
            print(f"   arg1: {arg1.shape}")
            print(f"   arg2: {arg2.shape}")
            
            # Tester si c'est x_embeddings et demographic_info séparés
            if len(arg1.shape) == 3 and len(arg2.shape) == 2:
                if arg1.shape[1:] == (18, 128) and arg2.shape[1] == 2:
                    print("   ✅ Format correct détecté!")
                    return self._forward_with_reconstructed(arg1, arg2)
        
        # Fallback: retourner quelque chose pour éviter le crash
        print("   ⚠️  FALLBACK: format non reconnu")
        return torch.zeros(args[0].shape[0], 9)  # [B, num_classes]
    
    def _forward_with_reconstructed(self, x_embeddings, demographic_info):
        """Forward avec les inputs reconstruits"""
        print(f"\n   🚀 Forward avec inputs reconstruits:")
        print(f"   x_embeddings: {x_embeddings.shape}")
        print(f"   demographic_info: {demographic_info.shape}")
        
        # Créer freq_bounds
        B, Freq, d = x_embeddings.shape
        device = x_embeddings.device
        
        freq_bounds = torch.tensor([
            [2., 4.], [4., 8.], [8., 10.], [10., 12.], [12., 18.],
            [18., 21.], [21., 30.], [30., 45.], [4., 30.]
        ], device=device, dtype=torch.float)
        
        if Freq == 18:
            freq_bounds = freq_bounds.repeat(2, 1)
        
        # Forward transformer
        x_coarse = self.shared_transformer(
            x_embeddings, freq_bounds, demographic_info,
            explanations=None, mode='standard'
        )
        
        logits = self.classifier_coarse(x_coarse.mean(dim=1))
        print(f"   ✅ Forward réussi: {logits.shape}")
        return logits


def test_explainer_debug():
    """Test détaillé de l'explainer"""
    
    print("🔍 DEBUG EXPLAINER CAPTUM")
    print("=" * 50)
    
    # Setup
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.model.num_classes = 9
    cfg.freeze()
    
    # Données
    with open("data/TDBRAIN/tokens/xai_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)
    
    # Modèle
    model = XaiGuiFormer(config=cfg, training_graphs=graphs[:5])
    
    # Données test
    loader = DataLoader(graphs[:2], batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    # Preprocessing (comme dans test_training.py)
    B = batch.y.shape[0]
    batch.x_tokens = batch.x_tokens.view(B, 18, 528)
    batch.freq_bounds = batch.freq_bounds.view(B, 18, 2)
    
    freq_bounds = batch.freq_bounds[0]
    age = batch.age.view(-1, 1)
    gender = batch.gender.view(-1, 1)
    
    # Forward pour obtenir x_embeddings
    x_embeddings = model.connectome_encoder(batch)
    demographic_info = torch.cat([age, gender], dim=1)
    
    print(f"📊 Données préparées:")
    print(f"   x_embeddings: {x_embeddings.shape}")
    print(f"   demographic_info: {demographic_info.shape}")
    print(f"   target: {batch.y}")
    
    # ✅ TEST 1: Créer wrapper debug
    wrapper = DebugWrapper(model)
    
    # ✅ TEST 2: Tester LayerDeepLift sur une couche
    print(f"\n🧪 TEST LayerDeepLift sur couche 0:")
    
    first_layer = model.shared_transformer.layers[0]
    explainer = LayerDeepLift(wrapper, first_layer)
    
    # Input tuple comme dans le code original
    inputs = (x_embeddings, demographic_info)
    target = batch.y
    baselines = (torch.zeros_like(x_embeddings), torch.zeros_like(demographic_info))
    
    print(f"   Inputs envoyés à LayerDeepLift:")
    print(f"   - type: {type(inputs)}")
    print(f"   - x_embeddings: {inputs[0].shape}")
    print(f"   - demographic_info: {inputs[1].shape}")
    print(f"   - target: {target}")
    
    try:
        attribution = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            attribute_to_layer_input=True
        )
        print(f"   ✅ Attribution réussie: {attribution.shape}")
        
    except Exception as e:
        print(f"   ❌ Attribution échouée: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 DIAGNOSTIC TERMINÉ")


if __name__ == "__main__":
    test_explainer_debug()
    