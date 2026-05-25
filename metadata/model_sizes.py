"""
Manual model size mappings for models without parseable size in name.

Sources: Model cards, papers, official documentation.
Verified 2025-12-30.

For MoE models, we use TOTAL params (not active) for bucketing,
since that's what users download and how models are marketed.
Active params noted in comments for reference.
"""

# Model ID -> size in billions (total params)
MANUAL_SIZES = {
    # =========================================================================
    # DeepSeek
    # =========================================================================
    # R1 family (MoE) - 671B total, 37B active
    # Note: HF shows 685B due to 14B MTP module included in weights
    'deepseek-ai/DeepSeek-R1': 671,
    'deepseek-ai/DeepSeek-R1-Zero': 671,
    'deepseek-ai/DeepSeek-R1-0528': 671,

    # V3 family (MoE) - 671B total, 37B active
    'deepseek-ai/DeepSeek-V3': 671,
    'deepseek-ai/DeepSeek-V3-0324': 671,
    'deepseek-ai/DeepSeek-V3-Base': 671,
    'deepseek-ai/DeepSeek-V3.1': 671,
    'deepseek-ai/DeepSeek-V3.1-Base': 671,
    'deepseek-ai/DeepSeek-V3.1-Terminus': 671,

    # V3.2 family (MoE) - 685B total (same arch as V3)
    'deepseek-ai/DeepSeek-V3.2': 685,
    'deepseek-ai/DeepSeek-V3.2-Exp': 685,
    'deepseek-ai/DeepSeek-V3.2-Exp-Base': 685,
    'deepseek-ai/DeepSeek-V3.2-Speciale': 685,

    # Math-V2 (MoE, same arch as V3.2) - 685B total
    'deepseek-ai/DeepSeek-Math-V2': 685,

    # OCR models (small dense)
    'deepseek-ai/DeepSeek-OCR': 3.3,
    'deepseek-ai/DeepSeek-OCR-2': 3.4,

    # V2 family (MoE) - 236B total, 21B active
    'deepseek-ai/DeepSeek-V2': 236,
    'deepseek-ai/DeepSeek-V2-Chat': 236,
    'deepseek-ai/DeepSeek-V2-Chat-0628': 236,
    'deepseek-ai/DeepSeek-V2.5': 236,
    'deepseek-ai/DeepSeek-V2.5-1210': 236,

    # V2-Lite (MoE) - 15.7B total, 2.4B active
    'deepseek-ai/DeepSeek-V2-Lite': 15.7,
    'deepseek-ai/DeepSeek-V2-Lite-Chat': 15.7,

    # Coder-V2 (MoE) - 236B total, 21B active
    'deepseek-ai/DeepSeek-Coder-V2-Base': 236,
    'deepseek-ai/DeepSeek-Coder-V2-Instruct': 236,
    'deepseek-ai/DeepSeek-Coder-V2-Instruct-0724': 236,

    # Coder-V2-Lite - 16B total, 2.4B active
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Base': 16,
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct': 16,

    # Prover - 7B dense
    'deepseek-ai/DeepSeek-Prover-V1': 7,
    'deepseek-ai/DeepSeek-Prover-V1.5-Base': 7,
    'deepseek-ai/DeepSeek-Prover-V1.5-RL': 7,
    'deepseek-ai/DeepSeek-Prover-V1.5-SFT': 7,

    # VL2 (vision-language, MoE backbone)
    'deepseek-ai/deepseek-vl2': 27,
    'deepseek-ai/deepseek-vl2-small': 16,
    'deepseek-ai/deepseek-vl2-tiny': 3,

    # Legacy coder
    'deepseek-ai/deepseek-coder-5.7bmqa-base': 5.7,

    # =========================================================================
    # Microsoft Phi
    # =========================================================================
    'microsoft/phi-1': 1.3,
    'microsoft/phi-1_5': 1.3,
    'microsoft/phi-2': 2.7,
    'microsoft/phi-2-pytdml': 2.7,
    'microsoft/phi-4': 14,

    # Phi-3 family
    'microsoft/Phi-3-mini-4k-instruct': 3.8,
    'microsoft/Phi-3-mini-128k-instruct': 3.8,
    'microsoft/Phi-3-small-8k-instruct': 7,
    'microsoft/Phi-3-small-128k-instruct': 7,
    'microsoft/Phi-3-medium-4k-instruct': 14,
    'microsoft/Phi-3-medium-128k-instruct': 14,
    'microsoft/Phi-3-vision-128k-instruct': 4.2,

    # Phi-3.5 family
    'microsoft/Phi-3.5-mini-instruct': 3.8,
    'microsoft/Phi-3.5-vision-instruct': 4.2,
    'microsoft/Phi-3.5-MoE-instruct': 42,  # 16x3.8B MoE

    # Phi-4 family
    'microsoft/Phi-4-mini-instruct': 3.8,
    'microsoft/Phi-4-mini-reasoning': 3.8,
    'microsoft/Phi-4-multimodal-instruct': 5.6,
    'microsoft/Phi-4-reasoning': 14,
    'microsoft/Phi-4-reasoning-plus': 14,

    # Phi MoE variants
    'microsoft/Phi-mini-MoE-instruct': 6.6,  # estimate
    'microsoft/Phi-tiny-MoE-instruct': 3.3,  # estimate

    # =========================================================================
    # Mistral
    # =========================================================================
    'mistralai/Mistral-Large-Instruct-2407': 123,
    'mistralai/Mistral-Large-Instruct-2411': 123,
    'mistralai/Mistral-Nemo-Base-2407': 12,
    'mistralai/Mistral-Nemo-Instruct-2407': 12,
    'mistralai/Mistral-Small-Instruct-2409': 22,
    'mistralai/Pixtral-Large-Instruct-2411': 124,
    'mistral-community/Pixtral-Large-Instruct-2411': 124,

    # Mixtral 8x7B (MoE) - 46.7B total, 12.9B active
    'mistralai/Mixtral-8x7B-v0.1': 46.7,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 46.7,

    # Mixtral 8x22B (MoE) - 140.6B total, 39B active
    'mistralai/Mixtral-8x22B-v0.1': 140.6,
    'mistralai/Mixtral-8x22B-Instruct-v0.1': 140.6,
    'mistral-community/Mixtral-8x22B-v0.1': 140.6,
    'mistral-community/Mixtral-8x22B-v0.1-original': 140.6,
    'mistral-community/mixtral-8x22B-Instruct-v0.3-original': 140.6,
    'mistral-community/mixtral-8x22B-v0.3': 140.6,
    'mistral-community/mixtral-8x22B-v0.3-original': 140.6,

    # Small 3 / Devstral / Magistral (24B)
    'mistralai/Devstral-Small-2505': 24,
    'mistralai/Magistral-Small-2506': 24,
    'mistralai/Magistral-Small-2507': 24,
    'mistralai/Magistral-Small-2509': 24,

    # =========================================================================
    # AI21 Jamba (MoE)
    # =========================================================================
    # Jamba-Large: 398B total, 94B active
    'ai21labs/AI21-Jamba-Large-1.5': 398,
    'ai21labs/AI21-Jamba-Large-1.6': 398,
    'ai21labs/AI21-Jamba-Large-1.7': 398,
    'ai21labs/AI21-Jamba-Large-1.7-FP8': 398,

    # Jamba-Mini: 52B total, 12B active
    'ai21labs/AI21-Jamba-Mini-1.5': 52,
    'ai21labs/AI21-Jamba-Mini-1.6': 52,
    'ai21labs/AI21-Jamba-Mini-1.7': 52,
    'ai21labs/AI21-Jamba-Mini-1.7-FP8': 52,
    'ai21labs/Jamba-v0.1': 52,

    # Tiny dev models
    'ai21labs/Jamba-tiny-dev': 0.5,  # estimate for dev model
    'ai21labs/Jamba-tiny-random': 0.5,

    # =========================================================================
    # MoonshotAI Kimi
    # =========================================================================
    # K2: 1T total (1040B), 32B active
    'moonshotai/Kimi-K2-Base': 1000,
    'moonshotai/Kimi-K2-Instruct': 1000,
    'moonshotai/Kimi-K2-Instruct-0905': 1000,
    'moonshotai/Kimi-K2-Thinking': 1000,

    # K2.5 (MoE) - 1T total
    'moonshotai/Kimi-K2.5': 1000,

    # Kimi-VL (MoE) - 16.4B total, 3B active
    'moonshotai/Kimi-VL-A3B-Instruct': 16.4,
    'moonshotai/Kimi-VL-A3B-Thinking': 16.4,
    'moonshotai/Kimi-VL-A3B-Thinking-2506': 16.4,

    # =========================================================================
    # Qwen (models without B in name)
    # =========================================================================
    'Qwen/Qwen-VL': 9.6,  # Based on Qwen-7B backbone + vision
    'Qwen/Qwen-VL-Chat': 9.6,

    # =========================================================================
    # Other orgs - estimates/lookups needed
    # =========================================================================
    # HuggingFaceTB SmolVLM (base versions without size in name)
    'HuggingFaceTB/SmolVLM-Base': 2,  # estimate
    'HuggingFaceTB/SmolVLM-Instruct': 2,
    'HuggingFaceTB/SmolVLM-Instruct-DPO': 2,
    'HuggingFaceTB/SmolVLM-Synthetic': 2,
    'HuggingFaceTB/SmolLM2-nanotron-ckpt': 0.135,
    'HuggingFaceTB/smolvlm-app-config': 0,  # not a model
    'HuggingFaceTB/stack-edu-classifier-python': 0.1,  # classifier

    # MiniMaxAI
    'MiniMaxAI/MiniMax-M1-40k': 456,  # MoE model
    'MiniMaxAI/MiniMax-M1-40k-hf': 456,
    'MiniMaxAI/MiniMax-M1-80k': 456,
    'MiniMaxAI/MiniMax-M1-80k-hf': 456,
    'MiniMaxAI/MiniMax-Text-01': 456,
    'MiniMaxAI/MiniMax-Text-01-hf': 456,
    'MiniMaxAI/MiniMax-VL-01': 456,  # Vision variant

    # MiniMax-M2 (MoE) - 229B total
    'MiniMaxAI/MiniMax-M2': 229,
    'MiniMaxAI/MiniMax-M2.1': 229,

    # Snowflake Arctic (MoE) - 480B total, 17B active
    'Snowflake/snowflake-arctic-base': 480,
    'Snowflake/snowflake-arctic-instruct': 480,
    'Snowflake/snowflake-arctic-instruct-vllm': 480,
    'Snowflake/e5-base-arctic-finetune': 0.1,  # embedding model

    # Skywork MoE
    'Skywork/Skywork-MoE-Base': 146,
    'Skywork/Skywork-MoE-Base-FP8': 146,
    'Skywork/R1V4': 30,  # 30B total, A3B active

    # Tencent Hunyuan
    'tencent/Tencent-Hunyuan-Large': 389,  # MoE, 52B active
    'tencent/Hunyuan-A13B-Instruct': 80,  # MoE, 13B active
    'tencent/Hunyuan-A13B-Pretrain': 80,
    'tencent/HunyuanOCR': 1,
    'tencent/POINTS-Reader': 7,  # estimate
    'tencent/Hunyuan3D-Omni': 7,

    # THUDM / zai-org (many are older/smaller models)
    'THUDM/glm-large-chinese': 0.335,
    'THUDM/glm-roberta-large': 0.355,
    'THUDM/cogvlm-chat-hf': 17,
    'THUDM/cogvlm-base-224-hf': 17,
    'THUDM/cogvlm-base-490-hf': 17,
    'THUDM/cogvlm-grounding-base-hf': 17,
    'THUDM/cogvlm-grounding-generalist-hf': 17,
    'THUDM/cogagent-chat-hf': 18,
    'THUDM/cogagent-vqa-hf': 18,
    'THUDM/cogvlm2-llama3-caption': 17,
    'THUDM/cogvlm2-video-llama3-base': 17,
    'THUDM/cogvlm2-video-llama3-chat': 17,
    'THUDM/CogAgent': 18,
    'THUDM/CogVLM': 17,
    'THUDM/WebGLM': 10,  # based on GLM-10B
    'THUDM/MathGLM': 7,
    'THUDM/MathGLM-Vision': 7,
    'THUDM/BPO': 7,
    'THUDM/MSAGPT': 7,
    'THUDM/glm-4-voice-decoder': 0.5,
    'THUDM/glm-4-voice-tokenizer': 0.1,

    # zai-org (mirrors of THUDM)
    'zai-org/glm-large-chinese': 0.335,
    'zai-org/glm-roberta-large': 0.355,
    'zai-org/cogvlm-chat-hf': 17,
    'zai-org/cogvlm-base-224-hf': 17,
    'zai-org/cogvlm-base-490-hf': 17,
    'zai-org/cogvlm-grounding-base-hf': 17,
    'zai-org/cogvlm-grounding-generalist-hf': 17,
    'zai-org/cogagent-chat-hf': 18,
    'zai-org/cogagent-vqa-hf': 18,
    'zai-org/cogvlm2-llama3-caption': 17,
    'zai-org/cogvlm2-video-llama3-base': 17,
    'zai-org/cogvlm2-video-llama3-chat': 17,
    'zai-org/CogAgent': 18,
    'zai-org/CogVLM': 17,
    'zai-org/WebGLM': 10,
    'zai-org/MathGLM': 7,
    'zai-org/MathGLM-Vision': 7,
    'zai-org/BPO': 7,
    'zai-org/MSAGPT': 7,
    'zai-org/glm-4-voice-decoder': 0.5,
    'zai-org/glm-4-voice-tokenizer': 0.1,
    'zai-org/GLM-4.5': 9,  # estimate
    'zai-org/GLM-4.5-Base': 9,
    'zai-org/GLM-4.5-Air': 9,
    'zai-org/GLM-4.5-Air-Base': 9,
    'zai-org/GLM-4.5V': 9,

    # GLM-4.6/4.7 family
    'zai-org/GLM-4.6': 357,  # MoE
    'zai-org/GLM-4.6-FP8': 357,
    'zai-org/GLM-4.6V': 108,  # MoE vision
    'zai-org/GLM-4.6V-FP8': 108,
    'zai-org/GLM-4.6V-Flash': 10,  # dense
    'zai-org/GLM-4.7': 358,  # MoE
    'zai-org/GLM-4.7-FP8': 358,
    'zai-org/GLM-4.7-Flash': 31,  # MoE lite

    # openbmb MiniCPM
    'openbmb/MiniCPM-V-4': 4.1,
    'openbmb/MiniCPM-MoE-8x2B': 14,  # MoE, 4B active
    'openbmb/Eurux-8x22b-kto': 141,  # Mixtral-based
    'openbmb/Eurux-8x22b-nca': 141,
    'openbmb/MiniCPM-V': 3,
    'openbmb/MiniCPM-V-2': 3,
    'openbmb/MiniCPM-V-2_6': 8,
    'openbmb/MiniCPM-V-4_5': 8,
    'openbmb/MiniCPM-o-2_6': 8,
    'openbmb/MiniCPM-Llama3-V-2_5': 8,
    'openbmb/MiniCPM4-MCP': 4,
    'openbmb/MiniCPM4-Survey': 4,
    'openbmb/MiniCPM-Embedding': 0.1,
    'openbmb/MiniCPM-Embedding-Light': 0.05,
    'openbmb/MiniCPM-Reranker': 0.4,
    'openbmb/MiniCPM-Reranker-Light': 0.1,
    'openbmb/VisCPM-Chat': 10,
    'openbmb/VisCPM-Paint': 10,
    'openbmb/VisRAG-Ret': 2,
    'openbmb/RLHF-V': 7,
    'openbmb/RLHF-V-SFT': 7,
    'openbmb/Ultra-FineWeb-classifier': 0.1,
    'openbmb/chattts_tokenizer': 0,  # not a model

    # nvidia MambaVision (vision models, relatively small)
    'nvidia/MambaVision-T-1K': 0.032,
    'nvidia/MambaVision-T2-1K': 0.035,
    'nvidia/MambaVision-S-1K': 0.050,
    'nvidia/MambaVision-B-1K': 0.098,
    'nvidia/MambaVision-B-21K': 0.098,
    'nvidia/MambaVision-L-1K': 0.228,
    'nvidia/MambaVision-L-21K': 0.228,
    'nvidia/MambaVision-L2-1K': 0.241,
    'nvidia/MambaVision-L2-512-21K': 0.241,
    'nvidia/MambaVision-L3-256-21K': 0.280,
    'nvidia/MambaVision-L3-512-21K': 0.280,

    # internlm
    'internlm/Intern-S1': 7,  # estimate
    'internlm/Intern-S1-mini': 3,
    'internlm/internlm-xcomposer2d5-clip': 0.3,
    'internlm/internlm2-step-prover': 7,
    'internlm/internlm2-wqx-vl-clip': 0.3,
    'internlm/internlm2_5-step-prover': 7,

    # ibm-granite
    'ibm-granite/granite-4.0-tiny-base-preview': 0.5,
    'ibm-granite/granite-4.0-tiny-preview': 0.5,
    'ibm-granite/granite-embedding-107m-multilingual': 0.107,
    'ibm-granite/granite-embedding-278m-multilingual': 0.278,
    'ibm-granite/granite-guardian-hap-125m': 0.125,
    'ibm-granite/granite-guardian-hap-38m': 0.038,

    # inclusionAI (various sizes, need verification)
    'inclusionAI/Ling-lite': 7,
    'inclusionAI/Ling-lite-base': 7,
    'inclusionAI/Ling-lite-1.5': 7,
    'inclusionAI/Ling-lite-1.5-2506': 7,
    'inclusionAI/Ling-lite-1.5-2507': 7,
    'inclusionAI/Ling-lite-base-1.5': 7,
    'inclusionAI/Ling-Coder-lite': 7,
    'inclusionAI/Ling-Coder-lite-base': 7,
    'inclusionAI/Ling-mini-2.0': 3,
    'inclusionAI/Ling-mini-base-2.0': 3,
    'inclusionAI/Ling-mini-base-2.0-5T': 3,
    'inclusionAI/Ling-mini-base-2.0-10T': 3,
    'inclusionAI/Ling-mini-base-2.0-15T': 3,
    'inclusionAI/Ling-mini-base-2.0-20T': 3,
    'inclusionAI/Ling-flash-2.0': 3,
    'inclusionAI/Ling-flash-2.0-GGUF': 3,
    'inclusionAI/Ling-flash-base-2.0': 3,
    'inclusionAI/Ling-plus': 32,
    'inclusionAI/Ling-plus-base': 32,
    'inclusionAI/Ring-lite': 7,
    'inclusionAI/Ring-lite-2506': 7,
    'inclusionAI/Ring-lite-2507': 7,
    'inclusionAI/Ring-lite-distill-preview': 7,
    'inclusionAI/Ring-lite-linear-preview': 7,
    'inclusionAI/Ring-mini-2.0': 3,
    'inclusionAI/Ring-mini-2.0-GGUF': 3,
    'inclusionAI/Ring-mini-linear-2.0': 3,
    'inclusionAI/Ring-flash-2.0': 3,
    'inclusionAI/Ring-flash-linear-2.0': 3,
    'inclusionAI/GroveMoE-Base': 100,  # MoE
    'inclusionAI/GroveMoE-Inst': 100,

    # inclusionAI 1T models (MoE) - ~1000B total
    'inclusionAI/Ling-1T': 1000,
    'inclusionAI/Ling-1T-FP8': 1000,
    'inclusionAI/Ring-1T': 1000,
    'inclusionAI/Ring-1T-FP8': 1000,
    'inclusionAI/Ring-1T-preview': 1000,
    'inclusionAI/Ring-1T-preview-FP8': 1000,
    'inclusionAI/M2-Reasoning': 7,
    'inclusionAI/Ming-Lite-Omni': 7,
    'inclusionAI/Ming-Lite-Omni-1.5': 7,
    'inclusionAI/Ming-Lite-Uni': 7,
    'inclusionAI/Rubicon-Preview': 7,
    'inclusionAI/ViLaSR': 7,

    # allenai
    'allenai/OLMo-Ladder-760M-0.5xC': 0.76,
    'allenai/FlexOlmo-7x7B-1T': 49,  # MoE, 7x7B
    'allenai/dolma2-tokenizer-U10F0F0': 0,  # tokenizer, not a model

    # Microsoft Florence-2
    'microsoft/Florence-2-base': 0.2,
    'microsoft/Florence-2-base-ft': 0.2,
    'microsoft/Florence-2-large': 0.8,
    'microsoft/Florence-2-large-ft': 0.8,

    # Qwen MoE
    'Qwen/Qwen1.5-MoE-A2.7B': 14.3,  # MoE, 2.7B active
    'Qwen/Qwen1.5-MoE-A2.7B-Chat': 14.3,

    # XiaomiMiMo (MoE) - 310B total
    'XiaomiMiMo/MiMo-V2-Flash': 310,
    'XiaomiMiMo/MiMo-V2-Flash-Base': 310,

    # vikhyatk / moondream
    'vikhyatk/moondream1': 1.9,
    'vikhyatk/moondream2': 1.9,
    'moondream/moondream3-preview': 9.3,

    # rednote-hilab
    'rednote-hilab/dots.ocr': 3,

    # llava-hf
    'llava-hf/bakLlava-v1-hf': 7,
}


def get_size_bucket(size_b):
    """
    Map parameter count to size bucket.

    Buckets:
    - <1B: tiny
    - 1-5B: small
    - 7-9B: 7B
    - 10-50B: medium
    - 50-100B: medium+
    - 100-250B: large
    - 250B+: giant
    """
    if size_b is None or size_b == 0:
        return None
    if size_b < 1:
        return '<1B'
    if size_b < 5:
        return '1-5B'
    if size_b < 10:
        return '7-9B'
    if size_b < 50:
        return '10-50B'
    if size_b < 100:
        return '50-100B'
    if size_b < 250:
        return '100-250B'
    return '250B+'


# Bucket order for display
BUCKET_ORDER = ['<1B', '1-5B', '7-9B', '10-50B', '50-100B', '100-250B', '250B+']


def parse_size_from_name(model_id):
    """
    Try to parse parameter count from model name.

    Handles patterns like:
    - Llama-3.1-8B-Instruct -> 8
    - Qwen2.5-72B -> 72
    - phi-3-mini-4k -> None (no size in name)
    - gemma-2b -> 2

    Returns size in billions, or None if not parseable.
    """
    import re

    model_name = model_id.split('/')[-1] if '/' in model_id else model_id

    # Pattern: number followed by B (case insensitive), not preceded by other letters
    # Handles: 8B, 72B, 0.5B, 1.5B, etc.
    patterns = [
        r'[-_](\d+\.?\d*)B[-_]',      # -8B- or _8B_
        r'[-_](\d+\.?\d*)B$',          # -8B at end
        r'^(\d+\.?\d*)B[-_]',          # 8B- at start
        r'[-_](\d+\.?\d*)b[-_]',       # lowercase
        r'[-_](\d+\.?\d*)b$',
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def get_model_size(model_id, safetensors_params=None):
    """
    Get model size in billions, using multiple resolution strategies.

    Priority:
    1. Manual mapping (MANUAL_SIZES) - most accurate for MoE, special cases
    2. Safetensors metadata (from models_dim.parquet)
    3. Parse from model name

    Args:
        model_id: Full model ID (e.g., "meta-llama/Llama-3.1-8B")
        safetensors_params: Optional total params from safetensors_parameters_json

    Returns:
        Size in billions, or None if unknown
    """
    # 1. Check manual mapping first (handles MoE, special cases)
    if model_id in MANUAL_SIZES:
        return MANUAL_SIZES[model_id]

    # 2. Use safetensors metadata if available
    if safetensors_params is not None and safetensors_params > 0:
        return safetensors_params / 1e9

    # 3. Parse from model name
    parsed = parse_size_from_name(model_id)
    if parsed is not None:
        return parsed

    return None


def resolve_model_sizes(df, params_col='safetensors_parameters_json'):
    """
    Add size_b and size_bucket columns to a DataFrame with model IDs.

    Args:
        df: DataFrame with 'id' column (model IDs)
        params_col: Column name for safetensors params JSON (or None to skip)

    Returns:
        DataFrame with added columns: size_b, size_bucket
    """
    from hf_utils import parse_safetensors_params

    df = df.copy()

    # Parse safetensors params if column exists
    if params_col and params_col in df.columns:
        df['_safetensors_total'] = df[params_col].apply(parse_safetensors_params)
    else:
        df['_safetensors_total'] = None

    # Resolve size using all strategies
    df['size_b'] = df.apply(
        lambda row: get_model_size(row['id'], row.get('_safetensors_total')),
        axis=1
    )

    # Map to buckets
    df['size_bucket'] = df['size_b'].apply(get_size_bucket)

    # Clean up temp column
    if '_safetensors_total' in df.columns:
        df = df.drop(columns=['_safetensors_total'])

    return df
