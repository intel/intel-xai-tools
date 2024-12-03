from modelgauge.suts.huggingface_client import HuggingFaceSUT, HuggingFaceToken
from modelgauge.secret_values import InjectSecret
from modelgauge.sut_registry import SUTS

HUGGING_FACE_REPO_ID = "Intel/neural-chat-7b-v3-3"
UNIQUE_ID = "neural-chat-7b-v3-3"
SUTS.register(HuggingFaceSUT, UNIQUE_ID, HUGGING_FACE_REPO_ID, InjectSecret(HuggingFaceToken))
