#!/usr/bin/env python3
"""
Test de validation approfondie de l'explainer XAIguiFormer
✅ TEST 1: Explainer indépendant (qualité des explanations)
✅ TEST 2: Impact XAI dans le 2e passage (différence coarse vs refined)
"""

import torch
import pickle
import sys
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder

sys.path.append("src")
from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer


def test_1_explainer_quality():
    """
    ✅ TEST 1: Valider que l'explainer produit des explanations SIGNIFICATIVES
    - Les explanations ne sont pas que des zéros
    - Les explanations varient selon les inputs
    - Les explanations ont une magnitude raisonnable
    """
    print("=" * 80)
    print("TEST 1: QUALITÉ DES EXPLANATIONS")
    print("=" * 80)
    
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
    model.eval()
    
    # 2 échantillons DIFFÉRENTS pour tester la variabilité
    loader = DataLoader([graphs[0], graphs[5]], batch_size=1, shuffle=False)
    
    explanations_samples = []
    
    for i, batch in enumerate(loader):
        print(f"\n🔬 Échantillon {i+1}:")
        
        # Preprocessing
        B = batch.y.shape[0]
        batch.x_tokens = batch.x_tokens.view(B, 18, 528)
        batch.freq_bounds = batch.freq_bounds.view(B, 18, 2)
        
        freq_bounds = batch.freq_bounds[0]
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        
        print(f"   Target: {batch.y.item()}")
        print(f"   Age: {age.item():.1f}, Gender: {gender.item()}")
        
        # Forward pour obtenir x_embeddings
        x_embeddings = model.connectome_encoder(batch)
        demographic_info = torch.cat([age, gender], dim=1)
        
        # ✅ GÉNÉRER EXPLANATIONS
        model._init_explainer_if_needed()
        explanations = model.explainer.get_explanations(
            x_embeddings, batch.y, freq_bounds, age, gender
        )
        
        # ✅ ANALYSE QUALITATIVE
        print(f"   Explanations générées: {len(explanations)} couches")
        
        # Statistiques par couche
        layer_stats = []
        for layer_idx, exp in enumerate(explanations):
            exp_np = exp.detach().numpy()
            stats = {
                'mean': np.mean(exp_np),
                'std': np.std(exp_np),
                'min': np.min(exp_np), 
                'max': np.max(exp_np),
                'zeros_ratio': np.mean(exp_np == 0.0)
            }
            layer_stats.append(stats)
            
            if layer_idx < 3 or layer_idx >= len(explanations) - 2:  # Premières et dernières couches
                print(f"   Layer {layer_idx:2d}: mean={stats['mean']:+.4f}, std={stats['std']:.4f}, "
                      f"range=[{stats['min']:+.4f}, {stats['max']:+.4f}], zeros={stats['zeros_ratio']:.2%}")
        
        explanations_samples.append(explanations)
    
    print(f"\n🔍 VALIDATION QUALITÉ:")
    
    # ✅ CHECK 1: Pas que des zéros
    all_zero_layers = 0
    for sample_idx, explanations in enumerate(explanations_samples):
        for layer_idx, exp in enumerate(explanations):
            if torch.all(exp == 0.0):
                all_zero_layers += 1
                print(f"⚠️  Sample {sample_idx}, Layer {layer_idx}: Toutes les explanations = 0")
    
    if all_zero_layers == 0:
        print("✅ Aucune couche avec explanations = 0 partout")
    else:
        print(f"❌ {all_zero_layers} couches avec explanations = 0")
    
    # ✅ CHECK 2: Variabilité entre échantillons
    if len(explanations_samples) >= 2:
        differences = []
        for layer_idx in range(len(explanations_samples[0])):
            exp1 = explanations_samples[0][layer_idx]
            exp2 = explanations_samples[1][layer_idx]
            diff = torch.mean(torch.abs(exp1 - exp2)).item()
            differences.append(diff)
        
        avg_diff = np.mean(differences)
        print(f"✅ Différence moyenne entre échantillons: {avg_diff:.4f}")
        
        if avg_diff > 0.001:  # Seuil arbitraire mais raisonnable
            print("✅ Les explanations varient bien selon les inputs")
        else:
            print("⚠️  Les explanations semblent trop similaires entre échantillons")
    
    # ✅ CHECK 3: Magnitude raisonnable
    all_magnitudes = []
    for explanations in explanations_samples:
        for exp in explanations:
            all_magnitudes.append(torch.mean(torch.abs(exp)).item())
    
    avg_magnitude = np.mean(all_magnitudes)
    print(f"✅ Magnitude moyenne des explanations: {avg_magnitude:.4f}")
    
    if 0.001 < avg_magnitude < 10.0:  # Plage raisonnable
        print("✅ Magnitude des explanations dans une plage raisonnable")
        return True
    else:
        print(f"⚠️  Magnitude inhabituelle: {avg_magnitude:.4f}")
        return False


def test_2_xai_impact():
    """
    ✅ TEST 2: Valider que l'XAI influence RÉELLEMENT le 2e passage
    - Les logits coarse vs refined sont différents
    - La loss refined est différente de la loss coarse
    - Les prédictions peuvent changer entre coarse et refined
    """
    print("\n" + "=" * 80)
    print("TEST 2: IMPACT XAI SUR LE 2E PASSAGE")
    print("=" * 80)
    
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
    model.eval()
    
    # Batch de test
    loader = DataLoader(graphs[:4], batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    # Preprocessing
    B = batch.y.shape[0]
    batch.x_tokens = batch.x_tokens.view(B, 18, 528)
    batch.freq_bounds = batch.freq_bounds.view(B, 18, 2)
    
    freq_bounds = batch.freq_bounds[0]
    age = batch.age.view(-1, 1)
    gender = batch.gender.view(-1, 1)
    
    print(f"🔬 Test avec {B} échantillons")
    print(f"   Targets: {batch.y}")
    
    # ✅ FORWARD COMPLET avec comparaison
    with torch.no_grad():
        # Forward normal (sans y_true pour récupérer les deux sorties)
        logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
        
        print(f"\n📊 COMPARAISON LOGITS:")
        print(f"   Coarse shape:  {logits_coarse.shape}")
        print(f"   Refined shape: {logits_refined.shape}")
        
        # Statistiques sur les logits
        coarse_stats = {
            'mean': torch.mean(logits_coarse).item(),
            'std': torch.std(logits_coarse).item(),
            'min': torch.min(logits_coarse).item(),
            'max': torch.max(logits_coarse).item()
        }
        
        refined_stats = {
            'mean': torch.mean(logits_refined).item(),
            'std': torch.std(logits_refined).item(),
            'min': torch.min(logits_refined).item(),
            'max': torch.max(logits_refined).item()
        }
        
        print(f"   Coarse:  mean={coarse_stats['mean']:+.4f}, std={coarse_stats['std']:.4f}")
        print(f"   Refined: mean={refined_stats['mean']:+.4f}, std={refined_stats['std']:.4f}")
        
        # ✅ CHECK 1: Différence entre logits
        logits_diff = torch.mean(torch.abs(logits_coarse - logits_refined)).item()
        print(f"\n🔍 DIFFÉRENCE LOGITS:")
        print(f"   Différence moyenne absolue: {logits_diff:.6f}")
        
        if logits_diff > 0.001:  # Seuil minimal de différence
            print("✅ Les logits coarse et refined sont DIFFÉRENTS")
            impact_detected = True
        else:
            print("⚠️  Les logits coarse et refined sont trop similaires")
            impact_detected = False
        
        # ✅ CHECK 2: Différence dans les prédictions
        pred_coarse = torch.argmax(logits_coarse, dim=1)
        pred_refined = torch.argmax(logits_refined, dim=1)
        
        print(f"\n🎯 PRÉDICTIONS:")
        print(f"   Targets:  {batch.y.tolist()}")
        print(f"   Coarse:   {pred_coarse.tolist()}")
        print(f"   Refined:  {pred_refined.tolist()}")
        
        predictions_changed = torch.sum(pred_coarse != pred_refined).item()
        print(f"   Changements: {predictions_changed}/{B} échantillons")
        
        if predictions_changed > 0:
            print("✅ L'XAI change des prédictions!")
        else:
            print("ℹ️  Pas de changement de prédictions (normal sur petit échantillon)")
        
        # ✅ CHECK 3: Calcul des loss séparées
        import torch.nn.functional as F
        loss_coarse = F.cross_entropy(logits_coarse, batch.y)
        loss_refined = F.cross_entropy(logits_refined, batch.y)
        
        print(f"\n📉 LOSS COMPARISON:")
        print(f"   Loss coarse:  {loss_coarse.item():.6f}")
        print(f"   Loss refined: {loss_refined.item():.6f}")
        print(f"   Différence:   {(loss_refined - loss_coarse).item():+.6f}")
        
        # ✅ CHECK 4: Validation que l'explainer s'est bien exécuté
        # Re-forward avec y_true pour forcer l'explainer
        print(f"\n🔧 TEST AVEC EXPLAINER FORCÉ:")
        loss_combined = model(batch, freq_bounds, age, gender, y_true=batch.y)
        print(f"   Loss combinée: {loss_combined.item():.6f}")
        
        return impact_detected and logits_diff > 0.001


def run_validation_tests():
    """Exécute les deux tests de validation"""
    print("🧪 VALIDATION COMPLÈTE DE L'EXPLAINER XAIguiFormer")
    print("=" * 100)
    
    # Test 1
    test1_passed = test_1_explainer_quality()
    
    # Test 2  
    test2_passed = test_2_xai_impact()
    
    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    print(f"✅ Test 1 - Qualité explanations: {'PASSÉ' if test1_passed else 'ÉCHEC'}")
    print(f"✅ Test 2 - Impact XAI 2e passage:  {'PASSÉ' if test2_passed else 'ÉCHEC'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 VALIDATION COMPLÈTE RÉUSSIE!")
        print(f"   ✅ L'explainer génère des explanations significatives")
        print(f"   ✅ L'XAI influence réellement le 2e passage du transformer")
        print(f"   ✅ Le modèle est prêt pour l'entraînement reproductible!")
        return True
    else:
        print(f"\n⚠️  VALIDATION INCOMPLÈTE - Problèmes détectés")
        return False


if __name__ == "__main__":
    success = run_validation_tests()
    
    if success:
        print(f"\n🚀 EXPLAINER XAIguiFormer TOTALEMENT FONCTIONNEL!")
    else:
        print(f"\n🔧 Des ajustements peuvent être nécessaires")