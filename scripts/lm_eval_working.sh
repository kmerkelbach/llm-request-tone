#!/usr/bin/env bash
LOGLEVEL=DEBUG \
lm_eval \
  --model local-chat-completions \
  --model_args "model=anthropic/claude-3.7-sonnet,base_url=https://openrouter.ai/api/v1/chat/completions,api_key=${OPENROUTER_API_KEY},num_concurrent=2,max_retries=3" \
  --tasks gsm8k,mmlu_pro \
  --num_fewshot 5 \
  --apply_chat_template \
  --batch_size 1 \
  --limit 1 \
  --gen_kwargs "temperature=0,max_tokens=512" \
  --output_path results_openrouter_smoke.json
