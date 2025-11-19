# Example configuration files showing how to use the new learning rate schedulers

# For Linear Learning Rate Decay (Auto-calculation enabled)
"""
# Linear decay from BASE_LR to END_FACTOR*BASE_LR
# TOTAL_ITERS will be automatically calculated based on training epochs and dataset size
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  MAX_EPOCH: 50           # Total training epochs
  LR_POLICY:
    TYPE: 'Linear'
    # TOTAL_ITERS: auto-calculated as (total_epochs × steps_per_epoch)
    START_FACTOR: 1.0     # Start with full BASE_LR
    END_FACTOR: 0.0       # End with 0 learning rate
    STEP_TYPE: 'Iter'     # Update every iteration
"""

# For Linear Learning Rate Decay (Manual setting)
"""
# Manual override of total iterations
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  LR_POLICY:
    TYPE: 'Linear'
    TOTAL_ITERS: 10000    # Manually set total iterations (overrides auto-calculation)
    START_FACTOR: 1.0     # Start with full BASE_LR
    END_FACTOR: 0.0       # End with 0 learning rate
    STEP_TYPE: 'Iter'     # Update every iteration
"""

# For Cosine Annealing Learning Rate (Auto-calculation enabled)
"""
# Cosine annealing from BASE_LR to ETA_MIN
# T_MAX will be automatically calculated based on training epochs and dataset size
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  MAX_EPOCH: 100          # Total training epochs
  LR_POLICY:
    TYPE: 'CosineAnnealing'
    # T_MAX: auto-calculated as (total_epochs × steps_per_epoch)
    ETA_MIN: 0.00001      # Minimum learning rate
    STEP_TYPE: 'Epoch'    # Update every epoch
"""

# For Cosine Annealing Learning Rate (Manual setting)
"""
# Manual override of T_MAX parameter
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  LR_POLICY:
    TYPE: 'CosineAnnealing'
    T_MAX: 5000           # Manually set period (overrides auto-calculation)
    ETA_MIN: 0.00001      # Minimum learning rate
    STEP_TYPE: 'Epoch'    # Update every epoch
"""

# For Cosine Annealing with Warm Restarts (Auto-calculation enabled)
"""
# Cosine annealing with periodic restarts
# T_0 will be automatically calculated as total_iterations / 4 if not specified
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  MAX_EPOCH: 80           # Total training epochs
  LR_POLICY:
    TYPE: 'CosineWarmRestarts'
    # T_0: auto-calculated as (total_epochs × steps_per_epoch) / 4
    T_MULT: 2             # Factor to increase T_i after restart
    ETA_MIN: 0.00001      # Minimum learning rate
    STEP_TYPE: 'Epoch'    # Update every epoch
"""

# For Cosine Annealing with Warm Restarts (Manual setting)
"""
# Manual override of T_0 parameter
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  LR_POLICY:
    TYPE: 'CosineWarmRestarts'
    T_0: 1000             # Manually set first restart period (overrides auto-calculation)
    T_MULT: 2             # Factor to increase T_i after restart
    ETA_MIN: 0.00001      # Minimum learning rate
    STEP_TYPE: 'Epoch'    # Update every epoch
"""

# Usage examples in different scenarios:

# 1. Fine-tuning scenario with linear decay (Auto-calculation)
"""
# Fine-tuning usually requires fewer epochs, auto-calculation handles this automatically
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.0001
  MAX_EPOCH: 20           # Shorter fine-tuning period
  LR_POLICY:
    TYPE: 'Linear'
    # TOTAL_ITERS: auto-calculated based on dataset and epochs
    START_FACTOR: 1.0
    END_FACTOR: 0.1       # Don't decay to 0, keep some learning
    STEP_TYPE: 'Iter'
"""

# 2. Long training with cosine annealing (Auto-calculation)
"""
# Long training benefits from smooth cosine decay, auto-calculated for full training period
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  MAX_EPOCH: 100          # Long training period
  LR_POLICY:
    TYPE: 'CosineAnnealing'
    # T_MAX: auto-calculated as (100 epochs × steps_per_epoch)
    ETA_MIN: 0.00001
    STEP_TYPE: 'Epoch'
"""

# 3. Training with periodic restarts (Auto-calculation)
"""
# Periodic restarts for better convergence, restart period auto-calculated
SOLVER:
  TYPE: 'ADAM'
  BASE_LR: 0.001
  MAX_EPOCH: 60           # Medium training period
  LR_POLICY:
    TYPE: 'CosineWarmRestarts'
    # T_0: auto-calculated as total_iterations / 4 (provides ~4 restart cycles)
    T_MULT: 1             # Keep same cycle length
    ETA_MIN: 0.00001
    STEP_TYPE: 'Epoch'
"""

# ========================================================================
# SUMMARY: Parameters that are now AUTO-CALCULATED (no longer need manual setting)
# ========================================================================

"""
With the new auto-calculation feature, the following parameters are automatically 
computed based on your training setup (MAX_EPOCH and dataset size):

1. Linear Scheduler:
   - TOTAL_ITERS: Calculated as (total_epochs × steps_per_epoch)
   
2. CosineAnnealing Scheduler:
   - T_MAX: Calculated as (total_epochs × steps_per_epoch)
   
3. CosineWarmRestarts Scheduler:
   - T_0: Calculated as (total_epochs × steps_per_epoch) / 4 (if not specified)

BENEFITS:
✅ No more manual calculation needed
✅ Automatically adapts to different datasets
✅ Scales with batch size changes
✅ Works with any number of training epochs

COMPATIBILITY:
- If you manually set these parameters in config, they will override auto-calculation
- All existing configurations continue to work without changes
- Other schedulers (Fix, Step, Plateau, Noam, MultiStep) are unaffected

SIMPLIFIED CONFIGURATION:
Instead of calculating iterations manually, just set:
- MAX_EPOCH: How many epochs you want to train
- The system handles the rest automatically!

DEPRECATED/OPTIONAL PARAMETERS:
❌ TOTAL_ITERS (for Linear scheduler) - now auto-calculated
❌ T_MAX (for CosineAnnealing scheduler) - now auto-calculated  
❌ T_0 (for CosineWarmRestarts scheduler) - now auto-calculated if not set
"""