import os
import os.path as osp
import numpy as np
import sys

from easydict import EasyDict as edict

# Ensure all imports (`lib.config` or `PureT.lib.config`) share the same module/state.
_this_module = sys.modules[__name__]
sys.modules.setdefault("PureT.lib.config", _this_module)
sys.modules.setdefault("lib.config", _this_module)

__C = edict()
# Consumers can get config by:
#   from lib.config import cfg
cfg = __C

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = edict()

# Minibatch size
__C.TRAIN.BATCH_SIZE = 10

# scheduled sampling
__C.TRAIN.SCHEDULED_SAMPLING = edict()

__C.TRAIN.SCHEDULED_SAMPLING.START = 0

__C.TRAIN.SCHEDULED_SAMPLING.INC_EVERY = 5

__C.TRAIN.SCHEDULED_SAMPLING.INC_PROB = 0.05

__C.TRAIN.SCHEDULED_SAMPLING.MAX_PROB = 0.25

# reinforcement learning
__C.TRAIN.REINFORCEMENT = edict()

__C.TRAIN.REINFORCEMENT.START = 30

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = edict()

# Minibatch size
__C.TEST.BATCH_SIZE = 18


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = edict()

# Data directory
__C.DATA_LOADER.NUM_WORKERS = 4

__C.DATA_LOADER.PIN_MEMORY = True

__C.DATA_LOADER.PERSISTENT_WORKERS = True

__C.DATA_LOADER.PREFETCH_FACTOR = 2

__C.DATA_LOADER.DROP_LAST = True

__C.DATA_LOADER.SHUFFLE = True

__C.DATA_LOADER.TRAIN_GV_FEAT = ''

__C.DATA_LOADER.TRAIN_ATT_FEATS = 'up_down_10_100'

__C.DATA_LOADER.VAL_GV_FEAT = ''

__C.DATA_LOADER.VAL_ATT_FEATS = 'up_down_10_100'

__C.DATA_LOADER.TEST_GV_FEAT = ''

__C.DATA_LOADER.TEST_ATT_FEATS = 'up_down_10_100'

__C.DATA_LOADER.TRAIN_ID = 'coco_train_image_id.txt'

__C.DATA_LOADER.VAL_ID = 'coco_val_image_id.txt'

__C.DATA_LOADER.TEST_ID = 'coco_test_image_id.txt'

__C.DATA_LOADER.TEST_4W_ID = 'coco_test4w_image_id.txt'

__C.DATA_LOADER.INPUT_SEQ_PATH = 'coco_train_input.pkl'

__C.DATA_LOADER.TARGET_SEQ_PATH = 'coco_train_target.pkl'

__C.DATA_LOADER.SEQ_PER_IMG = 5

__C.DATA_LOADER.MAX_FEAT = -1

# ---------------------------------------------------------------------------- #
# Corruption options (byte-stream level)
# ---------------------------------------------------------------------------- #
__C.CORRUPTION = edict()

# Active corruption types for byte streams (rbbf/rbsl/metadata_loss/none)
__C.CORRUPTION.BYTE_STREAM_TYPES = []

# Severity level shared across active corruption types
__C.CORRUPTION.BYTE_STREAM_LEVEL = 'S0'

# Optional overrides; negative values disable overrides
__C.CORRUPTION.RBBF = edict()
__C.CORRUPTION.RBBF.TRIGGER_PROB = -1.0
__C.CORRUPTION.RBBF.BURST_LAMBDA = -1.0
__C.CORRUPTION.RBBF.BIT_ERROR_RATE = -1.0

__C.CORRUPTION.RBSL = edict()
__C.CORRUPTION.RBSL.TRIGGER_PROB = -1.0
__C.CORRUPTION.RBSL.BURST_LAMBDA = -1.0
__C.CORRUPTION.RBSL.MAX_DROP_RATIO = -1.0

__C.CORRUPTION.METADATA = edict()
__C.CORRUPTION.METADATA.STRIP_APP_SEGMENTS = -1
__C.CORRUPTION.METADATA.ZERO_PREFIX_BYTES = -1
__C.CORRUPTION.METADATA.BODY_TRIM_RATIO = -1.0

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = edict()

__C.MODEL.TYPE = 'BLIP'               # PureT_byteformer/BLIP

__C.MODEL.SEQ_LEN = 17                  # include <EOS>/<BOS>

__C.MODEL.VOCAB_SIZE = 7375             # exclude <EOS>/<BOS> - Updated for Flickr8k

__C.MODEL.WORD_EMBED_DIM = 1000

__C.MODEL.WORD_EMBED_ACT = 'NONE'       # 'RELU', 'CELU', 'NONE'

__C.MODEL.WORD_EMBED_NORM = False

__C.MODEL.DROPOUT_WORD_EMBED = 0.0

__C.MODEL.GVFEAT_DIM = 2048

__C.MODEL.GVFEAT_EMBED_DIM = -1

__C.MODEL.GVFEAT_EMBED_ACT = 'NONE'     # 'RELU', 'CELU', 'NONE'

__C.MODEL.DROPOUT_GV_EMBED = 0.0

__C.MODEL.ATT_FEATS_DIM = 2048

__C.MODEL.ATT_FEATS_EMBED_DIM = -1

__C.MODEL.ATT_FEATS_EMBED_ACT = 'NONE'   # 'RELU', 'CELU', 'NONE'

__C.MODEL.DROPOUT_ATT_EMBED = 0.0

__C.MODEL.ATT_FEATS_NORM = False

__C.MODEL.ATT_HIDDEN_SIZE = 512

__C.MODEL.ATT_HIDDEN_DROP = 0.0

__C.MODEL.ATT_ACT = 'RELU'  # 'RELU', 'CELU', 'TANH'

__C.MODEL.RNN_SIZE = 1000

__C.MODEL.DROPOUT_LM = 0.5

# BOTTOM_UP
__C.MODEL.BOTTOM_UP = edict()

__C.MODEL.BOTTOM_UP.DROPOUT_FIRST_INPUT = 0.0

__C.MODEL.BOTTOM_UP.DROPOUT_SEC_INPUT = 0.0

# Transformer
__C.MODEL.TRANSFORMER = edict()

__C.MODEL.TRANSFORMER.PE_MAX_LEN = 5000


# Bilinear
__C.MODEL.BILINEAR = edict()

__C.MODEL.BILINEAR.DIM = -1

__C.MODEL.BILINEAR.ENCODE_ATT_MID_DIM = [1]

__C.MODEL.BILINEAR.DECODE_ATT_MID_DIM = [1]

__C.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT = 0.0

__C.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT = 0.0

__C.MODEL.BILINEAR.ATT_DIM = 1000

__C.MODEL.BILINEAR.ACT = 'RELU'  # 'RELU', 'CELU', 'TANH', 'GLU'

__C.MODEL.BILINEAR.ENCODE_DROPOUT = 0.1

__C.MODEL.BILINEAR.DECODE_DROPOUT = 0.1

__C.MODEL.BILINEAR.ENCODE_LAYERS = 1

__C.MODEL.BILINEAR.DECODE_LAYERS = 1

__C.MODEL.BILINEAR.TYPE = 'LowRank'

__C.MODEL.BILINEAR.ATTTYPE = 'SCAtt'

__C.MODEL.BILINEAR.HEAD = 8

__C.MODEL.BILINEAR.ENCODE_FF_DROPOUT = 0.1

__C.MODEL.BILINEAR.DECODE_FF_DROPOUT = 0.1

__C.MODEL.BILINEAR.ENCODE_BLOCK = 'LowRankBilinearEnc'

__C.MODEL.BILINEAR.DECODE_BLOCK = 'LowRankBilinearDec'

__C.MODEL.BILINEAR.ELU_ALPHA = 1.0

__C.MODEL.BILINEAR.BIFEAT_EMB_ACT = 'RELU'

__C.MODEL.BILINEAR.ENCODE_BIFEAT_EMB_DROPOUT = 0.3

__C.MODEL.BILINEAR.DECODE_BIFEAT_EMB_DROPOUT = 0.3

# HuggingFace/Transformer vision captioner options (used for BLIP and future models)
__C.MODEL.HF = edict()
__C.MODEL.HF.MODEL_ID = 'Salesforce/blip-image-captioning-base'  # HF repo id
__C.MODEL.HF.PROCESSOR_ID = ''  # Defaults to MODEL_ID when empty
__C.MODEL.HF.LOCAL_DIR = 'blip-image-captioning-base'  # Optional local cache dir
__C.MODEL.HF.DEVICE = 'cuda'  # auto: cuda if available else cpu
__C.MODEL.HF.SAFE_SERIALIZATION = True
__C.MODEL.HF.TRUST_REMOTE_CODE = False
__C.MODEL.HF.MIRROR = ''  # Optional HF mirror endpoint, e.g. https://hf-mirror.com
__C.MODEL.HF.DISABLE_PROXY = False  # Disable HTTP(S) proxy during HF downloads
__C.MODEL.HF.ALLOW_UNSAFE_TORCH_LOAD = False  # Allow .bin loading with torch<2.6 (security risk)
__C.MODEL.HF.TORCH_DTYPE = ''  # '', 'auto', 'float16', 'bfloat16'
__C.MODEL.HF.ATTN_IMPLEMENTATION = ''  # '', 'flash_attention_2' (if supported)
__C.MODEL.HF.LOW_CPU_MEM_USAGE = False
__C.MODEL.HF.GRADIENT_CHECKPOINTING = False
__C.MODEL.HF.PROMPT_SOURCE = ''  # '', 'openrouter', 'hf'
__C.MODEL.HF.SYSTEM_PROMPT = ''
__C.MODEL.HF.USER_PROMPT = ''
__C.MODEL.HF.PLACEHOLDER = 'A person standing in a room.'  # e.g. "A person standing in a room."
__C.MODEL.HF.USE_CHAT_TEMPLATE = False
__C.MODEL.HF.TRAINABLE = False  # Enable HF training path (e.g., LoRA finetuning)
__C.MODEL.HF.TRAIN_MODE = 'auto'  # auto/vision2seq/chat
__C.MODEL.HF.TRAIN_SYSTEM_PROMPT = ''
__C.MODEL.HF.TRAIN_USER_PROMPT = ''
__C.MODEL.HF.TRAIN_MAX_LENGTH = 128
__C.MODEL.HF.TRAIN_TRUNCATION = True
__C.MODEL.HF.TRAIN_LABEL_IGNORE = -100
__C.MODEL.HF.LORA = edict()
__C.MODEL.HF.LORA.ENABLED = False
__C.MODEL.HF.LORA.R = 8
__C.MODEL.HF.LORA.ALPHA = 16
__C.MODEL.HF.LORA.DROPOUT = 0.05
__C.MODEL.HF.LORA.BIAS = 'none'
__C.MODEL.HF.LORA.TASK_TYPE = 'CAUSAL_LM'
__C.MODEL.HF.LORA.TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
__C.MODEL.HF.LORA.MODULES_TO_SAVE = []
__C.MODEL.HF.LORA.SAVE_FULL_MODEL = False
__C.MODEL.HF.GENERATION = edict()
__C.MODEL.HF.GENERATION.MAX_LENGTH = 50
__C.MODEL.HF.GENERATION.MAX_NEW_TOKENS = -1
__C.MODEL.HF.GENERATION.NUM_BEAMS = 3

# OpenRouter vision captioning options (API-based models)
__C.MODEL.OPENROUTER = edict()
__C.MODEL.OPENROUTER.MODEL_ID = 'openai/gpt-5.1'
__C.MODEL.OPENROUTER.API_BASE = 'https://openrouter.ai/api/v1/chat/completions'
__C.MODEL.OPENROUTER.API_KEY = ''
__C.MODEL.OPENROUTER.API_KEY_PATH = 'openrouter_api'
__C.MODEL.OPENROUTER.HTTP_REFERER = ''
__C.MODEL.OPENROUTER.APP_TITLE = 'ByteCaption'
__C.MODEL.OPENROUTER.TIMEOUT = 60
__C.MODEL.OPENROUTER.MAX_TOKENS = 64
__C.MODEL.OPENROUTER.TEMPERATURE = 0.0
__C.MODEL.OPENROUTER.TOP_P = 1.0
__C.MODEL.OPENROUTER.MAX_WORKERS = 6
__C.MODEL.OPENROUTER.BATCH_SIZE = 1
__C.MODEL.OPENROUTER.PROXY = ''
__C.MODEL.OPENROUTER.SYSTEM_PROMPT = ''
__C.MODEL.OPENROUTER.USER_PROMPT = ''
__C.MODEL.OPENROUTER.IMAGE_DETAIL = ''  # e.g. "low" to reduce image tokens
__C.MODEL.OPENROUTER.FREQUENCY_PENALTY = None
__C.MODEL.OPENROUTER.PRESENCE_PENALTY = None
__C.MODEL.OPENROUTER.REPETITION_PENALTY = None
__C.MODEL.OPENROUTER.MIN_P = None
__C.MODEL.OPENROUTER.TOP_K = None
__C.MODEL.OPENROUTER.SEED = None
__C.MODEL.OPENROUTER.STOP = None
__C.MODEL.OPENROUTER.LOGPROBS = None
__C.MODEL.OPENROUTER.TOP_LOGPROBS = None
__C.MODEL.OPENROUTER.RESPONSE_FORMAT = None
__C.MODEL.OPENROUTER.EXTRA_HEADERS = None
__C.MODEL.OPENROUTER.EXTRA_PAYLOAD = None
# Reasoning control for OpenRouter models.
# Keep it as a dict-like structure so YAML can set e.g.:
#   REASONING: { enabled: false }
# or:
#   REASONING: { enabled: true, effort: "low" }
__C.MODEL.OPENROUTER.REASONING = edict()
__C.MODEL.OPENROUTER.REASONING.enabled = False
__C.MODEL.OPENROUTER.REASONING.effort = ''
__C.MODEL.OPENROUTER.PLACEHOLDER = 'A person standing in a room.'
__C.MODEL.OPENROUTER.RETRY = edict()
__C.MODEL.OPENROUTER.RETRY.MAX_ATTEMPTS = 3
__C.MODEL.OPENROUTER.RETRY.BACKOFF_BASE = 1.5
__C.MODEL.OPENROUTER.RETRY.BACKOFF_MAX = 20.0
__C.MODEL.OPENROUTER.RETRY.ON_EMPTY_RESPONSE = True
__C.MODEL.OPENROUTER.RETRY.ON_TRUNCATED = True
# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
__C.SOLVER = edict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.0005

# Solver type
__C.SOLVER.TYPE = 'ADAM'                 # 'ADAM', 'ADAMAX', 'SGD', 'ADAGRAD', 'RMSPROP', 'RADAM'

# Maximum number of SGD iterations
__C.SOLVER.MAX_EPOCH = 30

__C.SOLVER.MAX_ITER = 60000

__C.SOLVER.GRAD_CLIP = 0.1               # Norm:0.5 , Clamp:0.1

__C.SOLVER.GRAD_CLIP_TYPE = 'Clamp'      # 'Clamp', 'Norm'

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

__C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

__C.SOLVER.BIAS_LR_FACTOR = 2

__C.SOLVER.DISPLAY = 100

__C.SOLVER.TEST_INTERVAL = 1

__C.SOLVER.SNAPSHOT_ITERS = 3

__C.SOLVER.MAX_CHECKPOINTS = 2  # 最大保存的checkpoint数量，设置为-1则不限制

# SGD
__C.SOLVER.SGD = edict()
__C.SOLVER.SGD.MOMENTUM = 0.9

# ADAM
__C.SOLVER.ADAM = edict()
__C.SOLVER.ADAM.BETAS = [0.9, 0.999]
__C.SOLVER.ADAM.EPS = 1e-8

# LR_POLICY
# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = edict()
__C.SOLVER.LR_POLICY.TYPE = 'Fix'       # 'Fix', 'Step', 'Noam', 'Plateau'
__C.SOLVER.LR_POLICY.GAMMA = 0.8         # For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.LR_POLICY.STEP_SIZE = 3       # Uniform step size for 'steps' policy
__C.SOLVER.LR_POLICY.STEPS = (3,)        # Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs' policies
__C.SOLVER.LR_POLICY.STEP_TYPE = 'Epoch' # 'Epoch', 'Iter'

__C.SOLVER.LR_POLICY.WARMUP = 20000      # For Noam only
__C.SOLVER.LR_POLICY.FACTOR = 1.0        # For Noam only
__C.SOLVER.LR_POLICY.MODEL_SIZE = 1024   # For Noam only

__C.SOLVER.LR_POLICY.PLATEAU_FACTOR = 0.5
__C.SOLVER.LR_POLICY.PLATEAU_PATIENCE = 3

# Linear LR policy parameters
__C.SOLVER.LR_POLICY.TOTAL_ITERS = None    # [AUTO-CALCULATED] Total iterations for linear decay (set to number to override)
__C.SOLVER.LR_POLICY.START_FACTOR = 1.0    # Starting factor for linear decay
__C.SOLVER.LR_POLICY.END_FACTOR = 0.0      # Ending factor for linear decay

# Cosine Annealing parameters
__C.SOLVER.LR_POLICY.T_MAX = None          # [AUTO-CALCULATED] Maximum iterations for cosine annealing (set to number to override)
__C.SOLVER.LR_POLICY.ETA_MIN = 0.0           # Minimum learning rate

# Cosine Annealing with Warm Restarts parameters
__C.SOLVER.LR_POLICY.T_0 = None            # [AUTO-CALCULATED] Iterations for first restart (set to number to override)
__C.SOLVER.LR_POLICY.T_MULT = 2            # Factor increases T_i after a restart

# Warmup settings for cosine schedulers
__C.SOLVER.LR_POLICY.WARMUP_STEPS = 0      # Absolute warmup steps (linear ramp)
__C.SOLVER.LR_POLICY.WARMUP_RATIO = 0.0    # Warmup steps as ratio of total iters (used if WARMUP_STEPS==0)
__C.SOLVER.LR_POLICY.WARMUP_INIT_LR = 0.0  # Starting LR for warmup (defaults to 0)

# ---------------------------------------------------------------------------- #
# Losses options
# ---------------------------------------------------------------------------- #
__C.LOSSES = edict()

__C.LOSSES.XE_TYPE = 'CrossEntropy'      # 'CrossEntropy', 'LabelSmoothing'

__C.LOSSES.RL_TYPE = 'RewardCriterion'

__C.LOSSES.LABELSMOOTHING = 0.0

# ---------------------------------------------------------------------------- #
# SCORER options
# ---------------------------------------------------------------------------- #
__C.SCORER = edict()

__C.SCORER.TYPES = ['Cider']

__C.SCORER.WEIGHTS = [1.0]

__C.SCORER.GT_PATH = 'coco_train_gts.pkl'

__C.SCORER.CIDER_CACHED = 'coco_train_cider.pkl'

# ---------------------------------------------------------------------------- #
# PARAM options
# ---------------------------------------------------------------------------- #
__C.PARAM = edict()

__C.PARAM.WT = 'WT'

__C.PARAM.GLOBAL_FEAT = 'GV_FEAT'

__C.PARAM.ATT_FEATS = 'ATT_FEATS'

__C.PARAM.ATT_FEATS_MASK = 'ATT_FEATS_MASK'

__C.PARAM.P_ATT_FEATS = 'P_ATT_FEATS'

__C.PARAM.STATE = 'STATE'

__C.PARAM.INPUT_SENT = 'INPUT_SENT'

__C.PARAM.TARGET_SENT = 'TARGET_SENT'

__C.PARAM.INDICES = 'INDICES'

# ---------------------------------------------------------------------------- #
# Inference options
# ---------------------------------------------------------------------------- #
__C.INFERENCE = edict()

__C.INFERENCE.VOCAB = 'coco_vocabulary.txt'

__C.INFERENCE.ID_KEY = 'image_id'

__C.INFERENCE.CAP_KEY = 'caption'

__C.INFERENCE.EVAL = 'COCO'

__C.INFERENCE.VAL_ANNFILE = 'captions_val5k.json'

__C.INFERENCE.TEST_ANNFILE = 'captions_test5k.json'

__C.INFERENCE.BEAM_SIZE = 1

__C.INFERENCE.GREEDY_DECODE = True # Greedy decode or sample decode

__C.INFERENCE.COCO_PATH = './coco-caption'

__C.INFERENCE.SAVE_EVAL_IMAGES_DIR = None

__C.INFERENCE.SAVE_EVAL_IMAGES_MAX = 0

__C.INFERENCE.SAMPLE_PREVIEW = 5

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# Logger name
__C.LOGGER_NAME = 'log'

# Image Mean
__C.MEAN = [0.485, 0.456, 0.406]

# Image std
__C.STD = [0.229, 0.224, 0.225]

__C.SEED = -1

__C.TEMP_DIR = './data/temp'

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    #for k, v in a.iteritems(): python2
    for k, v in a.items(): # python3
        # a must specify keys that are in b
        #if not b.has_key(k):
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r', encoding='utf-8') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader ))

    _merge_a_into_b(yaml_cfg, __C)
