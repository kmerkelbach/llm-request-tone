#!/usr/bin/env bash
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
LOGLEVEL=DEBUG \
lm_eval \
  --model local-chat-completions \
  --model_args "model=deepseek/deepseek-r1-distill-llama-8b,base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=2,max_retries=3" \
  --tasks gsm8k,mmlu_pro \
  --num_fewshot 5 \
  --apply_chat_template \
  --batch_size 1 \
  --limit 1 \
  --gen_kwargs "temperature=0,max_tokens=512" \
  --output_path results_openrouter_smoke.json
