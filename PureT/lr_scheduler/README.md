# Learning Rate Schedulers Documentation

This document describes the newly added learning rate schedulers: Linear, CosineAnnealing, and CosineWarmRestarts.

## Available Schedulers

### 1. Linear Learning Rate Scheduler (`Linear`)

Linearly decreases the learning rate from `start_factor * base_lr` to `end_factor * base_lr` over `total_iters` iterations.

**Configuration Parameters:**
- `TOTAL_ITERS`: Total number of iterations for the decay
- `START_FACTOR`: Starting multiplier for base learning rate (default: 1.0)
- `END_FACTOR`: Ending multiplier for base learning rate (default: 0.0)

**Use Cases:**
- Fine-tuning pretrained models
- Training with limited iterations
- Gradual learning rate reduction

**Example Configuration:**
```yaml
SOLVER:
  LR_POLICY:
    TYPE: 'Linear'
    TOTAL_ITERS: 10000
    START_FACTOR: 1.0
    END_FACTOR: 0.1
    SETP_TYPE: 'Iter'
```

### 2. Cosine Annealing Learning Rate Scheduler (`CosineAnnealing`)

Anneals the learning rate using a cosine function from `base_lr` to `eta_min` over `T_max` epochs/iterations.

**Configuration Parameters:**
- `T_MAX`: Maximum number of epochs/iterations for one cosine cycle
- `ETA_MIN`: Minimum learning rate (default: 0)

**Use Cases:**
- Long training runs
- Smooth learning rate decay
- Better convergence properties than step decay

**Example Configuration:**
```yaml
SOLVER:
  LR_POLICY:
    TYPE: 'CosineAnnealing'
    T_MAX: 50
    ETA_MIN: 0.00001
    SETP_TYPE: 'Epoch'
```

### 3. Cosine Annealing with Warm Restarts (`CosineWarmRestarts`)

Implements cosine annealing with periodic "warm restarts" where the learning rate is reset to the initial value.

**Configuration Parameters:**
- `T_0`: Number of epochs/iterations for the first restart period
- `T_MULT`: Factor to increase the period after each restart (default: 1)
- `ETA_MIN`: Minimum learning rate (default: 0)

**Use Cases:**
- Escaping local minima
- Better exploration during training
- Training with multiple convergence cycles

**Example Configuration:**
```yaml
SOLVER:
  LR_POLICY:
    TYPE: 'CosineWarmRestarts'
    T_0: 20
    T_MULT: 2
    ETA_MIN: 0.00001
    SETP_TYPE: 'Epoch'
```

## Complete Configuration Examples

### Example 1: Linear Decay for Fine-tuning
```yaml
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.0005
  LR_POLICY:
    TYPE: 'Linear'
    TOTAL_ITERS: 5000
    START_FACTOR: 1.0
    END_FACTOR: 0.2
    SETP_TYPE: 'Iter'
```

### Example 2: Cosine Annealing for Long Training
```yaml
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  LR_POLICY:
    TYPE: 'CosineAnnealing'
    T_MAX: 100
    ETA_MIN: 0.00001
    SETP_TYPE: 'Epoch'
```

### Example 3: Warm Restarts for Better Convergence
```yaml
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  LR_POLICY:
    TYPE: 'CosineWarmRestarts'
    T_0: 10
    T_MULT: 2
    ETA_MIN: 0.00001
    SETP_TYPE: 'Epoch'
```

## Migration from Existing Schedulers

### From Fixed LR:
```yaml
# Before
SOLVER:
  LR_POLICY:
    TYPE: 'Fix'

# After (with gradual decay)
SOLVER:
  LR_POLICY:
    TYPE: 'Linear'
    TOTAL_ITERS: 10000
    START_FACTOR: 1.0
    END_FACTOR: 0.5
    SETP_TYPE: 'Iter'
```

### From Step LR:
```yaml
# Before
SOLVER:
  LR_POLICY:
    TYPE: 'Step'
    STEP_SIZE: 10
    GAMMA: 0.5

# After (smoother decay)
SOLVER:
  LR_POLICY:
    TYPE: 'CosineAnnealing'
    T_MAX: 50
    ETA_MIN: 0.00001
    SETP_TYPE: 'Epoch'
```

## Usage with Command Line

You can use these schedulers by modifying your configuration file:

```bash
# Example training command
cd /root/autodl-tmp/ByteCaption && PYTHONPATH=/root/autodl-tmp/ByteCaption python PureT/main.py \
  --folder PureT/experiments/ByteCaption_XE \
  --dataset coco \
  --eval_steps 300 \
  --val_samples 50
```

The scheduler will be automatically applied based on your configuration file settings.

## Testing

To test the schedulers, run:
```bash
cd /root/autodl-tmp/ByteCaption/PureT/lr_scheduler
python test_schedulers.py
```

This will generate a plot showing the learning rate curves for all schedulers.

## Implementation Notes

1. All schedulers support both `'Iter'` and `'Epoch'` step types via `SETP_TYPE`
2. The schedulers are compatible with all existing optimizers (Adam, SGD, etc.)
3. Learning rate curves can be monitored via wandb logging
4. The implementation follows PyTorch's scheduler interface for consistency