# Batch API Guide for Dataset Generation

This guide explains how to use OpenAI's Batch API to generate your dataset efficiently and cost-effectively.

## Why Use Batch API?

- **50% cost savings** compared to real-time API
- **More efficient** - generate many candidates, then validate locally
- **Better for large datasets** - ideal for 1,500+ scenarios
- **No need for retry logic** - validation happens after generation

## Complete Workflow

### Step 1: Create Batch Request Files

Generate JSONL files containing batch requests:

```bash
python scripts/create_batch_requests.py \
  --task refusal \
  --num-batches 10 \
  --scenarios-per-request 10
```

**What this does:**
- Creates 10 batch files in `requests/refusal/`
- Each file contains 50 API requests
- Each request asks for 10 scenario pairs
- Total: 500 requests × 10 pairs = 5,000 candidate pairs

**Options:**
- `--task`: Task name (default: `refusal`)
- `--num-batches`: Number of batch files (default: `10`)
- `--scenarios-per-request`: Scenarios per API call (default: `10`)
- `--output-dir`: Output directory (default: `requests`)

**Cost estimate:** ~$2.50-5.00 for 500 requests (with 50% Batch API discount)

### Step 2: Submit Batch Jobs

Upload and submit the batch files to OpenAI:

```bash
python scripts/submit_batches.py --task refusal
```

**What this does:**
- Uploads all batch files from `requests/refusal/`
- Creates batch jobs on OpenAI's servers
- Saves batch metadata to `requests/refusal/batch_metadata.json`
- Returns batch IDs for tracking

**Output:**
```
Submitted 10 batches
Batch IDs saved to: requests/refusal/batch_metadata.json
```

### Step 3: Check Batch Status

Monitor the progress of your batch jobs:

```bash
python scripts/check_batch_status.py --task refusal
```

**What this does:**
- Checks status of all submitted batches
- Shows completion progress
- Updates metadata file

**Example output:**
```
request_batch0.jsonl: completed
  ✓ Completed: 50/50

request_batch1.jsonl: in_progress
  ⏳ Progress: 35/50
```

**Batch statuses:**
- `validating`: OpenAI is validating your request file
- `in_progress`: Requests are being processed
- `finalizing`: Almost done
- `completed`: Ready to download
- `failed`: Something went wrong

**Typical completion time:** 1-24 hours (usually 2-6 hours)

You can check status multiple times. Run this periodically until all batches show `completed`.

### Step 4: Download Results

Once batches are completed, download the results:

```bash
python scripts/download_batch_results.py --task refusal
```

**What this does:**
- Downloads output files for all completed batches
- Saves to `batch_outputs/refusal/`
- Skips already-downloaded files

**Output:**
```
Downloaded: 10
Output directory: batch_outputs/refusal/
```

### Step 5: Process and Validate

Process the batch results and validate with tokenizers:

```bash
python scripts/process_batch_results.py \
  --task refusal \
  --output dataset.jsonl \
  --target-scenarios 1500
```

**What this does:**
- Loads all batch output files
- Parses GPT-4o responses (handles various formats)
- Validates each scenario with Qwen1.5, Llama-3.1, and SOLAR tokenizers
- Keeps only scenarios that pass ALL validation checks
- Stops when target number of valid scenarios is reached
- Writes final dataset to `dataset.jsonl`

**Validation checks:**
1. Equal token counts across all tokenizers
2. Exactly one token difference across all tokenizers
3. Both single-token and multi-token variants must pass

**Options:**
- `--task`: Task name
- `--input-dir`: Directory with batch outputs (default: `batch_outputs`)
- `--output`: Output JSONL file (default: `dataset.jsonl`)
- `--target-scenarios`: Stop after N valid scenarios (default: `1500`)
- `--cache-dir`: Tokenizer cache directory

**Example output:**
```
Processing Complete
================================================================================
Total API responses processed: 500
Total pairs generated: 4,523
Valid scenarios: 1,500
Validation failures: 3,023
Success rate: 33.2%

Total prompts written: 6,000
Output file: dataset.jsonl
================================================================================
```

## Complete Example

Here's the full workflow from start to finish:

```bash
# 1. Create batch request files
python scripts/create_batch_requests.py --num-batches 10

# 2. Submit to OpenAI
python scripts/submit_batches.py

# 3. Wait and check status (repeat until completed)
python scripts/check_batch_status.py

# 4. Download results when ready
python scripts/download_batch_results.py

# 5. Process and validate
python scripts/process_batch_results.py --target-scenarios 1500 --output dataset.jsonl

# 6. Verify the dataset
python scripts/example_usage.py dataset.jsonl
```

## Understanding Success Rates

The token-level constraints are strict, so expect:
- **~30-40% success rate** for scenarios passing validation
- This is normal and expected!
- That's why we generate 5,000 candidates to get 1,500 valid ones

The Batch API makes this efficient because:
- Generation is cheap (50% discount)
- Validation is free (done locally)
- No wasted retry API calls

## File Structure

After running all steps, you'll have:

```
dataset_generation/
├── requests/
│   └── refusal/
│       ├── request_batch0.jsonl          # Batch request files
│       ├── request_batch1.jsonl
│       ├── ...
│       └── batch_metadata.json           # Tracking info
├── batch_outputs/
│   └── refusal/
│       ├── output_batch0.jsonl           # Raw GPT-4o responses
│       ├── output_batch1.jsonl
│       └── ...
└── dataset.jsonl                         # Final validated dataset
```

## Advantages vs Real-Time API

| Aspect | Real-Time API | Batch API |
|--------|---------------|-----------|
| Cost | Full price | 50% discount |
| Speed | Immediate | 1-24 hours |
| Retries | Manual | Not needed |
| Validation | During generation | After generation |
| Efficiency | Generate & validate each | Generate all, then validate |
| Best for | <100 scenarios | 1,500+ scenarios |

For 1,500 scenarios:
- **Real-time**: ~$40-50 (with retries)
- **Batch API**: ~$5-10 (no retries needed)

## Troubleshooting

### Batch status shows "failed"
- Check the batch details on OpenAI dashboard
- Ensure your request file format is correct
- Try resubmitting that specific batch

### Low success rate (<20%)
- Check the refusal task configuration in `config/tasks/refusal.yaml`
- The generation instructions may need tuning
- Consider adding more examples to guide GPT-4o

### No output_file_id in metadata
- Batch may still be processing
- Run `check_batch_status.py` to refresh
- Wait and try again

### "No batch output files found"
- Run `download_batch_results.py` first
- Check that batches have completed
- Verify the correct task name

## Tips for Best Results

1. **Generate more than you need**: Request 3-5x your target to account for validation failures
2. **Process incrementally**: You can process batches as they complete, don't need to wait for all
3. **Monitor costs**: Check your OpenAI usage dashboard periodically
4. **Save metadata**: Don't delete `batch_metadata.json` until you're done

## Cost Breakdown (for 1,500 scenarios)

1. **Batch requests**: 500 requests × $0.001 = ~$0.50 (input)
2. **Batch responses**: 500 responses × $0.005 = ~$2.50 (output)  
3. **Batch API discount**: 50% off = **~$1.50 total**
4. **Tokenizer validation**: Free (local)

**Total cost: ~$1.50-3.00** (vs $40-50 with real-time API)

## Next Steps

After generating your dataset:
1. Validate it: `python scripts/validate_dataset.py dataset.jsonl`
2. Inspect it: `python scripts/example_usage.py dataset.jsonl`
3. Use it for your interpretability research!

## Questions?

See the main [README.md](README.md) for more information about the project.

