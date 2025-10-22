# Dataset Generation Workflows

This document compares the two available workflows for generating your interpretability dataset.

## Quick Comparison

| Feature | Real-Time API | Batch API (Recommended) |
|---------|---------------|-------------------------|
| **Cost for 1,500 scenarios** | $40-50 | $1.50-3.00 |
| **Time to complete** | 1-2 hours | 2-6 hours (mostly waiting) |
| **Complexity** | Simple, one command | Multi-step process |
| **Efficiency** | Generate & validate each | Generate all, filter later |
| **Best for** | Quick tests (<100) | Production (1,500+) |
| **API calls needed** | Many (with retries) | Fixed number |
| **Recommended** | Testing only | Production use ✓ |

## Workflow 1: Real-Time API (Original)

**Use case:** Quick testing with small datasets (<100 scenarios)

### Single Command

```bash
python scripts/generate_dataset.py \
  --num-scenarios 10 \
  --output test.jsonl
```

### How It Works

1. Generates one scenario at a time
2. Validates immediately with tokenizers
3. Retries up to 5 times if validation fails
4. Continues until target is reached

### Pros
- Simple, one command
- Immediate results
- Good for testing

### Cons
- Expensive (many API calls due to retries)
- Slow validation rate (~20-30% success)
- Can take many API calls to get valid scenarios

### Full Options

```bash
python scripts/generate_dataset.py \
  --task refusal \
  --num-scenarios 1500 \
  --output dataset.jsonl \
  --model gpt-4o \
  --checkpoint checkpoint.json \
  --cache-dir .cache \
  --log-level INFO
```

## Workflow 2: Batch API (Recommended)

**Use case:** Production datasets (1,500+ scenarios), cost-effective generation

### Five-Step Process

```bash
# Step 1: Create batch request files (10 files, 500 total requests)
python scripts/create_batch_requests.py --num-batches 10

# Step 2: Submit to OpenAI Batch API
python scripts/submit_batches.py

# Step 3: Check status (repeat until completed)
python scripts/check_batch_status.py

# Step 4: Download results when ready
python scripts/download_batch_results.py

# Step 5: Process and validate locally
python scripts/process_batch_results.py --target-scenarios 1500
```

### How It Works

1. **Create requests**: Generate JSONL files with batch API requests
2. **Submit**: Upload files to OpenAI, get batch IDs
3. **Wait**: Batches process in 1-24 hours (usually 2-6)
4. **Download**: Retrieve completed results
5. **Validate**: Run local tokenizer validation, keep valid scenarios

### Pros
- **50% cost savings** (Batch API discount)
- **More efficient**: Generate many candidates, filter locally
- **Better success rate**: No wasted retry API calls
- **Scalable**: Can generate 10,000+ candidates easily

### Cons
- Multi-step process
- Requires waiting for batch completion
- More complex workflow

### Detailed Guide

See [BATCH_API_GUIDE.md](BATCH_API_GUIDE.md) for complete instructions.

## Which Should I Use?

### Use Real-Time API if you:
- Need immediate results
- Are testing with <100 scenarios
- Don't mind higher costs
- Want simplicity

### Use Batch API if you:
- Need 1,500+ scenarios (production dataset)
- Want to minimize costs (50% discount)
- Can wait a few hours for results
- Want maximum efficiency

## Recommendation

For your research project requiring 1,500 scenarios:
👉 **Use Batch API** - You'll save ~$40-45 and it's the standard approach for production datasets.

The real-time API is great for testing (e.g., generate 10 scenarios to verify everything works), then switch to Batch API for the full dataset.

## Example: Testing First, Then Production

```bash
# Phase 1: Quick test with real-time API (5 scenarios)
python scripts/generate_dataset.py \
  --num-scenarios 5 \
  --output test.jsonl \
  --log-level DEBUG

# Verify the test dataset
python scripts/example_usage.py test.jsonl

# Phase 2: Full production with Batch API (1,500 scenarios)
python scripts/create_batch_requests.py --num-batches 10
python scripts/submit_batches.py
# ... wait for completion ...
python scripts/process_batch_results.py --target-scenarios 1500 --output dataset.jsonl

# Verify the production dataset
python scripts/validate_dataset.py dataset.jsonl
```

## Cost Comparison (Real Example)

For generating 1,500 valid scenarios:

### Real-Time API
- Need ~10,000 API calls (due to 15% success rate with retries)
- Input tokens: ~3,000,000 @ $0.0025/1K = $7.50
- Output tokens: ~2,000,000 @ $0.01/1K = $20.00
- **Total: ~$27.50 - $50** (depending on retry rate)

### Batch API
- 500 API calls (fixed, no retries)
- Input tokens: ~150,000 @ $0.00125/1K = $0.19 (50% discount)
- Output tokens: ~100,000 @ $0.005/1K = $0.50 (50% discount)
- Validation: Free (local)
- **Total: ~$1.50 - $3.00**

**Savings: ~$25-47** (90-95% reduction)

## Technical Details

### Validation Process

Both workflows use the same validation:
1. Check equal token counts across Qwen1.5, Llama-3.1, and SOLAR
2. Check exactly one token difference
3. Both single-token and multi-token variants must pass

The difference is **when** validation happens:
- **Real-time**: After each API call (immediate retry)
- **Batch**: After all API calls complete (no retries)

### Success Rates

Typical success rates for passing validation:
- **Single scenario**: 15-20% (why real-time needs retries)
- **Batch filtering**: 30-40% (no regeneration bias)

The batch approach gets better success rates because there's no iterative regeneration that might introduce patterns the tokenizers don't like.

## Questions?

- **Batch API details**: See [BATCH_API_GUIDE.md](BATCH_API_GUIDE.md)
- **General setup**: See [README.md](README.md)
- **Quick start**: See [QUICKSTART.md](QUICKSTART.md)

