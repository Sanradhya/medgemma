"""Medical Image Analyzer — Streamlit front-end."""

import os

from dotenv import load_dotenv
from PIL import Image
import streamlit as st

from dicom_utils import dicom_to_pil_image
from engine import generate_report, resolve_inference_mode, resolve_provider

load_dotenv()

st.set_page_config(
    page_title="Medical Image Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #f4f7fb; }
        [data-testid="stSidebar"] {
            background-color: #0f2c4c;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #e8eef5 !important;
        }
        .main-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f2c4c;
            margin-bottom: 0.15rem;
        }
        .subtitle {
            color: #5a6f86;
            font-size: 0.95rem;
            margin-bottom: 1.4rem;
        }
        .disclaimer {
            font-size: 0.78rem;
            color: #7a8b9c;
            margin-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_INSTRUCTIONS = (
    "Draft a structured preliminary radiological finding report "
    "emphasizing visible anomalies."
)
MODEL_OPTIONS = ["google/medgemma-4b-it"]
SCAN_TYPES = ["X-Ray", "Ultrasound", "MRI"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "dcm"]
PROVIDER_LABELS = {
    "modal": "MedGemma on Modal (GPU on demand)",
    "openai_compatible": "Other hosted MedGemma (Vertex / vLLM)",
    "gemini": "Gemini (not MedGemma)",
    "openai": "OpenAI vision (not MedGemma)",
}

if "review_image" not in st.session_state:
    st.session_state.review_image = None
if "review_filename" not in st.session_state:
    st.session_state.review_filename = ""
if "draft_report" not in st.session_state:
    st.session_state.draft_report = ""
if "finalized" not in st.session_state:
    st.session_state.finalized = False


def load_uploaded_image(uploaded_file) -> Image.Image:
    """Load a PNG/JPEG as PIL, or a DICOM as pixels-only RGB (no PHI)."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".dcm"):
        return dicom_to_pil_image(uploaded_file)
    return Image.open(uploaded_file).convert("RGB")


try:
    default_mode = resolve_inference_mode()
except ValueError:
    default_mode = "api"
default_provider = resolve_provider()
if default_provider not in PROVIDER_LABELS:
    default_provider = "modal"

with st.sidebar:
    st.markdown("### Settings")
    st.caption(
        "This laptop does not load MedGemma. Analyze starts a Modal GPU, then the GPU "
        "shuts down after a short idle window."
    )
    inference_mode = st.radio(
        "Inference mode",
        options=["api", "local"],
        index=0 if default_mode == "api" else 1,
        format_func=lambda value: (
            "Cloud API (recommended)" if value == "api" else "Local GPU (downloads MedGemma)"
        ),
        help="Cloud API works on typical laptops. Local mode needs a GPU with enough VRAM.",
    )

    selected_model = MODEL_OPTIONS[0]
    provider = default_provider
    api_key = ""
    api_url = os.getenv("MEDGEMMA_API_URL") or os.getenv("OPENAI_BASE_URL") or ""
    hf_token = ""

    if inference_mode == "api":
        provider_keys = list(PROVIDER_LABELS.keys())
        provider = st.selectbox(
            "Cloud provider",
            options=provider_keys,
            index=provider_keys.index(default_provider),
            format_func=lambda key: PROVIDER_LABELS[key],
        )
        if provider == "modal":
            api_url = st.text_input(
                "Modal endpoint URL",
                value=api_url,
                help="URL printed by `modal deploy modal_app.py` (*.modal.run).",
            )
            env_modal_key = os.getenv("MEDGEMMA_API_KEY") or ""
            api_key = st.text_input(
                "Modal API key (optional)",
                value="",
                type="password",
                placeholder="Using MEDGEMMA_API_KEY" if env_modal_key else "Optional shared secret",
            )
            if not api_key and env_modal_key:
                st.caption("Using MEDGEMMA_API_KEY from the environment.")
            st.caption(
                "The first Analyze after idle can take several minutes while Modal "
                "starts a GPU and loads MedGemma. Later calls in the next ~2 minutes are faster."
            )
        elif provider == "gemini":
            env_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            api_key = st.text_input(
                "Gemini API key",
                value="",
                type="password",
                placeholder="Using GEMINI_API_KEY" if env_gemini else "AIza...",
                help="Create a key in Google AI Studio. Image pixels are sent to Google.",
            )
            if not api_key and env_gemini:
                st.caption("Using GEMINI_API_KEY from the environment.")
            st.markdown("[Get a Gemini API key](https://aistudio.google.com/apikey)")
        elif provider == "openai":
            env_openai = os.getenv("OPENAI_API_KEY") or ""
            api_key = st.text_input(
                "OpenAI API key",
                value="",
                type="password",
                placeholder="Using OPENAI_API_KEY" if env_openai else "sk-...",
            )
            if not api_key and env_openai:
                st.caption("Using OPENAI_API_KEY from the environment.")
        else:
            api_url = st.text_input(
                "MedGemma endpoint URL",
                value=api_url,
                help="OpenAI-compatible chat completions URL for a warm GPU server.",
            )
            api_key = st.text_input(
                "Endpoint API key (optional)",
                value="",
                type="password",
            )
            selected_model = st.selectbox(
                "Model selection",
                options=MODEL_OPTIONS,
                index=0,
            )
        st.caption("Uploaded pixels are sent to the selected API. DICOM headers are not.")
    else:
        selected_model = st.selectbox(
            "Model selection",
            options=MODEL_OPTIONS,
            index=0,
            help="Vision-language model loaded onto this machine.",
        )
        env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or ""
        hf_token = st.text_input(
            "Hugging Face token",
            value="",
            type="password",
            help=(
                "MedGemma is a gated model. Accept the license on Hugging Face, "
                "then paste a token with access. You can also set HF_TOKEN."
            ),
            placeholder="hf_..." if not env_token else "Using HF_TOKEN from environment",
        )
        if not hf_token and env_token:
            st.caption("Using Hugging Face token from the environment.")
        st.markdown(
            "[Request access to google/medgemma-4b-it]"
            "(https://huggingface.co/google/medgemma-4b-it)"
        )
        st.caption("Local mode downloads ~8 GB and needs a GPU. Typical laptops will be very slow.")

    st.divider()
    if inference_mode == "api":
        st.caption(f"Active path: cloud `{provider}`")
    else:
        st.caption(f"Active model: `{selected_model}` (local)")
    st.caption("For research and decision-support prototyping only.")

st.markdown('<p class="main-title">Medical Image Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a diagnostic image, provide clinical context, '
    "and generate a structured preliminary finding report.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload medical image",
    type=IMAGE_TYPES,
    help="Accepted formats: PNG, JPG, JPEG, DCM (DICOM). DICOM headers are not used.",
)

scan_type = st.selectbox("Scan type", options=SCAN_TYPES, index=0)

clinician_instructions = st.text_area(
    "Clinician instructions / patient context",
    value=DEFAULT_INSTRUCTIONS,
    height=120,
    help="Add specific questions, history, or reporting preferences.",
)

analyze = st.button("Analyze Image", type="primary", use_container_width=False)

if analyze:
    if uploaded_file is None:
        st.warning("Please upload a PNG, JPG, JPEG, or DICOM (.dcm) file before analyzing.")
    else:
        try:
            image = load_uploaded_image(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            st.stop()

        prompt = (
            f"Scan type: {scan_type}. "
            f"{clinician_instructions.strip()}"
        )
        try:
            if inference_mode != "api":
                spinner = "Analyzing image locally — this can take a long time without a GPU..."
            elif provider == "modal":
                spinner = (
                    "Calling MedGemma on Modal — first request after idle can take several minutes..."
                )
            else:
                spinner = "Sending image to the analysis API..."
            with st.spinner(spinner):
                resolved_token = (
                    hf_token
                    or os.getenv("HF_TOKEN")
                    or os.getenv("HUGGING_FACE_HUB_TOKEN")
                    or ""
                ).strip()
                if resolved_token:
                    os.environ["HF_TOKEN"] = resolved_token
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved_token
                report = generate_report(
                    image,
                    prompt,
                    model_id=selected_model,
                    hf_token=resolved_token or None,
                    inference_mode=inference_mode,
                    provider=provider,
                    api_key=api_key or None,
                    api_url=api_url or None,
                )
        except Exception as exc:
            message = str(exc).lower()
            if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in message:
                st.error(
                    "The local model ran out of GPU memory. Use Cloud API mode, "
                    "or run on a machine with more VRAM."
                )
            elif "gated" in message or "401" in message or "restricted" in message:
                st.error(
                    "The model or API rejected this request (unauthorized or gated)."
                )
                st.markdown(
                    "For Modal, check `MEDGEMMA_API_URL` and that secret `huggingface` "
                    "has an HF token with MedGemma access. For local mode, accept the "
                    "license on [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)."
                )
            elif isinstance(exc, PermissionError):
                st.error(str(exc))
            else:
                st.error(f"Analysis failed: {exc}")
            st.stop()

        st.session_state.review_image = image
        st.session_state.review_filename = uploaded_file.name
        st.session_state.draft_report = report
        st.session_state.finalized = False

if st.session_state.review_image is not None:
    image_col, report_col = st.columns(2, gap="large")
    with image_col:
        st.subheader("Uploaded image")
        caption = st.session_state.review_filename or "Uploaded study"
        if caption.lower().endswith(".dcm"):
            caption = f"{caption} (pixels only; PHI stripped)"
        st.image(
            st.session_state.review_image,
            caption=caption,
            use_container_width=True,
        )
    with report_col:
        st.text_area(
            "Draft Report (Edit to Finalize)",
            height=420,
            key="draft_report",
            help="Review and edit the model draft before approving.",
        )
        if st.button("Approve & Finalize", type="primary"):
            st.session_state.finalized = True

        if st.session_state.finalized:
            st.success(
                "Report approved and finalized by the reviewing clinician. "
                "This draft is not a signed diagnostic report."
            )

st.markdown(
    '<p class="disclaimer">This tool produces a preliminary, non-diagnostic '
    "draft. Findings must be reviewed and signed by a qualified clinician. "
    "Do not use as a substitute for professional medical judgment. "
    "DICOM files are converted from pixel data only; study metadata and PHI "
    "are not sent to the model. In cloud mode, image pixels are sent to the "
    "configured API.</p>",
    unsafe_allow_html=True,
)
