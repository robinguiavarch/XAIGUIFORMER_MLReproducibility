from yacs.config import CfgNode as CN

_C = CN()

########################################################################
# basic settings
########################################################################
_C.model_name = 'XAIguiFormer'
_C.dataset = 'TDBRAIN' 
_C.root = 'data/TDBRAIN'
_C.out_root = 'outputs/'
_C.num_workers = 4
_C.print_freq = 1
_C.pretrain = False
_C.pretrained_model = ''
_C.seed = 42  # ✅ AJOUTÉ : reproducibilité

########################################################################
# preprocessing settings
########################################################################
_C.preprocessing = CN()
_C.preprocessing.n_workers = 4
_C.preprocessing.n_jobs = None
_C.preprocessing.l_freq = 1
_C.preprocessing.h_freq = 45
_C.preprocessing.line_frequency = 50 

_C.preprocessing.path = CN()
_C.preprocessing.path.electrode_locations = 'data/TDBRAIN/elec_loc.csv'
_C.preprocessing.path.data_dir = 'data/TDBRAIN/raw'
_C.preprocessing.path.save_dir = 'data/TDBRAIN/preprocessed'

_C.preprocessing.ica = CN()
_C.preprocessing.ica.n_components = 20
_C.preprocessing.ica.method = 'infomax'
_C.preprocessing.ica.fit_params = CN()
_C.preprocessing.ica.fit_params.extended = True
_C.preprocessing.ica.ic_label_threshold = 0.0

######################################################
# frequency band range and methods of constructing FC
######################################################
_C.connectome = CN()
_C.connectome.frequency_band = CN()
_C.connectome.frequency_band.delta = [2., 4.]
_C.connectome.frequency_band.theta = [4., 8.]
_C.connectome.frequency_band.low_alpha = [8., 10.]
_C.connectome.frequency_band.high_alpha = [10., 12.]
_C.connectome.frequency_band.low_beta = [12., 18.]
_C.connectome.frequency_band.mid_beta = [18., 21.]
_C.connectome.frequency_band.high_beta = [21., 30.]
_C.connectome.frequency_band.gamma = [30., 45.]
_C.connectome.frequency_band.beta = [12., 30.] 
_C.connectome.frequency_band.theta_beta_ratio = [4., 30.]  # ✅ AJOUTÉ : θ/β ratio (très important selon l'article)

_C.connectome.methods = ['coh', 'wpli']

_C.connectome.path = CN()
_C.connectome.path.data_dir = 'data/TDBRAIN/preprocessed'
_C.connectome.path.save_dir = 'data/TDBRAIN/connectome'

######################################################
# model settings (✅ CORRIGÉS selon l'article Table 9)
######################################################
_C.model = CN()
_C.model.num_node_feat = 26  # TDBRAIN a 26 canaux
_C.model.num_edge_feat = 1
_C.model.num_classes = 3  # ADHD, MDD, OCD
_C.model.num_head = 4  # ✅ Conforme à l'article
_C.model.dim_node_feat = 128  # ✅ Conforme à l'article
_C.model.dim_edge_feat = 128
_C.model.gnn_type = 'GINEConv'
_C.model.num_gnn_layer = 4  # ✅ Conforme à l'article
_C.model.num_transformer_layer = 12  # ✅ CORRIGÉ : l'article utilise 12 couches
_C.model.explainer_type = 'DeepLift'
_C.model.dropout = 0.1  # ✅ CORRIGÉ : l'article utilise dropout=0.1, pas 0.0
_C.model.mlp_ratio = 4.
_C.model.init_values = 1e-3  # ✅ AJOUTÉ : layer scale init selon l'article
_C.model.attn_drop = 0.
_C.model.droppath = 0.


_C.model.token_input_dim = 528  # ✅ NOUVEAU: dimension réelle des x_tokens  
_C.model.estimated_electrodes = 34  # ✅ NOUVEAU: basé sur diagnostic

# ✅ AJOUTS POUR ABLATION STUDY
_C.model.use_xai_guidance = True       
_C.model.use_drofe = True                
_C.model.use_demographics = True  

######################################################
# train settings (✅ CORRIGÉS selon Table 9 de l'article)
######################################################
_C.train = CN()
_C.train.batch_size = 64  # ✅ Conforme à l'article pour TDBRAIN

_C.train.epochs = 100

_C.train.optimizer = CN()
_C.train.optimizer.eps = 1e-8
_C.train.optimizer.betas = (0.9, 0.99)
_C.train.optimizer.lr = 5e-5  # ✅ CORRIGÉ : l'article utilise 5e-5 pour TDBRAIN, pas 1e-4
_C.train.optimizer.weight_decay = 1e-5  # ✅ CORRIGÉ : l'article utilise 1e-5, pas 0.01

_C.train.lr_scheduler = CN()
_C.train.lr_scheduler.lr_min = 1e-7
_C.train.lr_scheduler.warmup_epochs = 5
_C.train.lr_scheduler.warmup_lr = 1e-7  # ✅ CORRIGÉ : cohérent avec lr_min

_C.train.criterion = CN()
_C.train.criterion.smoothing = 0.1
_C.train.criterion.alpha = 0.7  # ✅ CORRIGÉ : l'article utilise 0.7, pas 1.0 !!!

######################################################
# augmentation settings
######################################################
_C.aug = CN()
_C.aug.mixup = 1.0  # ✅ AJUSTÉ : l'article utilise G-Mixup
_C.aug.cutmix = 0.5  # ✅ AJUSTÉ : l'article utilise G-CutMix
_C.aug.cutmix_minmax = None
_C.aug.mixup_prob = 1.0
_C.aug.mixup_switch_prob = 0.5
_C.aug.mixup_mode = 'batch'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()