"""model configuration"""

import os

BOOL_LANGCHAIN_LOG = True

# LLM model configuration
DEFAULT_MAX_LEN = 32000
API_MAX_TRY_NUM = 3
CURRENT_LLM_MODEL = os.environ.get("CURRENT_LLM_MODEL_ENV", "ERNIE-Speed-128K")

# if CURRENT_LLM_MODEL not in llm_model, use the following config
DEFAULT_LLM_MODEL_API_KEY = os.environ.get("DEFAULT_LLM_MODEL_API_KEY_ENV", "None")
DEFAULT_LLM_MODEL_API_BASE_URL = os.environ.get(
    "DEFAULT_LLM_MODEL_API_BASE_URL_ENV", ""
)


DEFAULT_LLM_MODEL_MODEL_NAME = os.environ.get("DEFAULT_LLM_MODEL_MODEL_NAME_ENV", "")
DEFAULT_LLM_MODEL_MAX_LEN = int(
    os.environ.get("DEFAULT_LLM_MODEL_MAX_LEN_ENV", DEFAULT_MAX_LEN)
)

# list of llm models
llm_model = {
    "glm-4": {
        "API_KEY": "0ef6cf494dd27e2dd6aea2f571c050ce.1XR0Wcoj7FEqzy1Y",
        "API_BASE_URL": "https://open.bigmodel.cn/api/paas/v4/",
        "MODEL": "glm-4",
        "MAX_LEN": DEFAULT_MAX_LEN,
    },
    "qwen-72b-chat": {
        "API_KEY": "sk-ba610f0899144927a07463ffebacf6e2",
        "API_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "MODEL": "qwen-72b-chat",
        "MAX_LEN": DEFAULT_MAX_LEN,
    },
    "chatglm3-6b": {
        "API_KEY": "None",
        "API_BASE_URL": "http://172.16.0.129:8004/v1",
        "MODEL": "chatglm3-6b",
        "MAX_LEN": DEFAULT_MAX_LEN,
    },
    "kimi-moonshot": {
        "API_KEY": "sk-bdgRNhFMCqoBDkYBRYTVAllNcay0Fsg9x58ILyB0D2OrSYGE",
        "API_BASE_URL": "https://api.moonshot.cn/v1",
        "MODEL": "moonshot-v1-32k",
        "MAX_LEN": DEFAULT_MAX_LEN,
    },
    "chatglm3-6b-train": {
        "API_KEY": "None",
        "API_BASE_URL": "http://172.16.0.132:8000/v1",
        # "MODEL": "train_2024-07-12-06-06-47",
        "MODEL": "train_2024-07-26-06-39-53",
        "MAX_LEN": DEFAULT_MAX_LEN,
    },
    "local_32k": {
        "API_KEY": "None",
        "API_BASE_URL": "http://172.16.0.132:8000/v1",
        # "MODEL": "train_2024-07-12-06-06-47",
        "MODEL": "train_2024-07-26-06-39-53",
        "MAX_LEN": DEFAULT_MAX_LEN,
    },
    "ERNIE-Speed-128K": {
        "QIANFAN_AK": "xL76zFBUhnzHijj8KfyTgTt6",
        "QIANFAN_SK": "8gRs5aMgKOk4wcp2lS9SOwGr2rym7Ie7",
        "MODEL": "ERNIE-Speed-128K",
        "MAX_LEN": 128000,
    },
}
