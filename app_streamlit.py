import os
import gc
import json
import glob
import streamlit as st
import torch
import requests
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download, login, HfApi, list_models, snapshot_download
from PIL import Image

# ============== 環境變數載入 ==============
load_dotenv()  # 載入 .env 檔案

def get_secret(key, default=""):
    """從 st.secrets 或環境變數取得設定值"""
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)

# ============== 設定 ==============
MODEL_CACHE_DIR = "./models"
LORA_CACHE_DIR = "./loras"
OUTPUT_DIR = "./outputs"
HISTORY_DIR = "./history"
HF_TOKEN = get_secret("HF_TOKEN", "")
CIVIT_TOKEN = get_secret("CIVIT_TOKEN", "")
DEFAULT_STEPS = int(get_secret("DEFAULT_STEPS", "20"))
DEFAULT_CFG = float(get_secret("DEFAULT_CFG", "7.0"))
MAX_HISTORY_IMAGES = int(get_secret("MAX_HISTORY_IMAGES", "100"))

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(LORA_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ============== 預設模型 ==============
PRESET_MODELS = {
    "BK-SDM-Tiny (極速輕量 1.5)": "nota-ai/bk-sdm-tiny",
    "Stable Diffusion v1.5 (通用)": "runwayml/stable-diffusion-v1-5",
    "Dreamlike Anime 1.0 (動漫)": "dreamlike-art/dreamlike-anime-1.0",
    "Kernel NSFW (寫實/成人)": "Kernel/sd-nsfw",
    "Realistic Vision V5.1 (高畫質寫實)": "SG161222/Realistic_Vision_V5.1_noVAE",
    "SDXL 1.0 Base (高畫質底模)": "stabilityai/stable-diffusion-xl-base-1.0",
}

HF_FILE_MODELS = {
    "SDXL Lightning (極速 SDXL)": ("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"),
    "Pony Diffusion XL V6 (動漫/成人)": ("PonyXL_v6", "ponyxl_v6.safetensors"),
}

# ============== 記憶體管理 ==============
def clear_memory():
    """清理記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def optimize_pipeline(pipe):
    """優化管線以適應 CPU/低記憶體環境"""
    # 啟用注意力切片以減少記憶體使用
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
    # 啟用 VAE 切片
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    return pipe

# ============== Session State 初始化 ==============
def init_session_state():
    """初始化所有 session state 變數"""
    if "pipe" not in st.session_state:
        st.session_state.pipe = None
    if "current_model_path" not in st.session_state:
        st.session_state.current_model_path = ""
    if "current_model_is_sdxl" not in st.session_state:
        st.session_state.current_model_is_sdxl = False
    if "active_loras" not in st.session_state:
        st.session_state.active_loras = {}
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = HF_TOKEN
    if "civit_token" not in st.session_state:
        st.session_state.civit_token = CIVIT_TOKEN
    if "hf_search_results" not in st.session_state:
        st.session_state.hf_search_results = []
    if "history_loaded" not in st.session_state:
        st.session_state.history_loaded = False

# ============== 圖片歷史記錄 ==============
def save_to_history(image, prompt, neg_prompt, seed, steps, cfg, width, height, model_path):
    """儲存圖片到歷史記錄"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_seed{seed}.png"
    filepath = os.path.join(HISTORY_DIR, filename)
    
    # 儲存圖片
    image.save(filepath)
    
    # 儲存元資料
    metadata = {
        "timestamp": timestamp,
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "model": model_path,
        "filename": filename
    }
    
    json_path = filepath.replace(".png", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return filepath

def load_history():
    """載入歷史記錄"""
    history = []
    json_files = sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json")), reverse=True)
    
    for json_file in json_files[:MAX_HISTORY_IMAGES]:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            image_path = json_file.replace(".json", ".png")
            if os.path.exists(image_path):
                metadata["image_path"] = image_path
                history.append(metadata)
        except:
            continue
    
    return history

# ============== HuggingFace 模型搜尋 ==============
def search_hf_models(query, limit=10):
    """搜尋 HuggingFace 上的 Stable Diffusion 模型"""
    try:
        api = HfApi()
        models = list_models(
            search=query,
            filter="text-to-image",
            limit=limit,
            token=st.session_state.hf_token if st.session_state.hf_token else None
        )
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags if model.tags else []
            })
        return results
    except Exception as e:
        st.error(f"搜尋失敗: {str(e)}")
        return []

def download_hf_model(model_id):
    """下載 HuggingFace 模型到本地快取"""
    try:
        with st.spinner(f"下載模型 {model_id} 中..."):
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=MODEL_CACHE_DIR,
                token=st.session_state.hf_token if st.session_state.hf_token else None
            )
        return f"✅ 模型已下載至: {local_path}\n您可以在「本地模型」中載入此模型"
    except Exception as e:
        return f"❌ 下載失敗: {str(e)}"

# ============== 下載函數 ==============
def download_and_backup(url, folder, civit_token="", hf_token=""):
    """下載檔案並備份"""
    headers = {}
    if "civitai.com" in url and civit_token:
        headers["Authorization"] = f"Bearer {civit_token}"
    
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    
    # 嘗試從 header 取得檔名
    content_disp = response.headers.get("content-disposition", "")
    filename = "downloaded_model.safetensors"
    if "filename=" in content_disp:
        filename = content_disp.split("filename=")[1].strip('"')
    
    local_filepath = os.path.join(folder, filename)
    
    # 下載進度
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = st.progress(0, text=f"下載中: {filename}")
    
    with open(local_filepath, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(progress, text=f"下載中: {filename} ({int(progress*100)}%)")
    
    progress_bar.empty()
    return local_filepath, filename, "✅ 下載完成"

# ============== 載入模型 ==============
@st.cache_resource
def load_pipeline_cached(model_source, is_local_file=False, hf_token=""):
    """快取模型載入"""
    is_sdxl = False
    
    if is_local_file:
        if "sdxl" in model_source.lower() or "xl" in model_source.lower():
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_source,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            is_sdxl = True
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                model_source,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
    else:
        # 判斷是否為 SDXL
        if "sdxl" in model_source.lower() or "xl" in model_source.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_source,
                torch_dtype=torch.float32,
                use_auth_token=hf_token if hf_token else None,
            )
            is_sdxl = True
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_source,
                torch_dtype=torch.float32,
                use_auth_token=hf_token if hf_token else None,
            )
    
    pipe.to("cpu")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    # 優化管線以適應 CPU/低記憶體環境
    pipe = optimize_pipeline(pipe)

    return pipe, model_source, is_sdxl

def load_pipeline(model_source, is_local_file=False):
    """載入模型管線"""
    try:
        pipe, path, is_sdxl = load_pipeline_cached(
            model_source, 
            is_local_file, 
            st.session_state.hf_token
        )
        st.session_state.pipe = pipe
        st.session_state.current_model_path = path
        st.session_state.current_model_is_sdxl = is_sdxl
        st.session_state.active_loras = {}
        return f"✅ 模型載入成功: {model_source}"
    except Exception as e:
        return f"❌ 載入失敗: {str(e)}"

# ============== 模型處理 ==============
def handle_model_dropdown(choice):
    """處理預設模型選擇"""
    if not choice:
        return "請選擇模型"
    
    model_id = PRESET_MODELS.get(choice)
    if model_id:
        return load_pipeline(model_id)
    return "❌ 未知的模型選擇"

def handle_hf_file_model(choice):
    """處理 HF 檔案模型"""
    if not choice:
        return "請選擇模型"
    
    repo_id, filename = HF_FILE_MODELS.get(choice, (None, None))
    if repo_id and filename:
        try:
            filepath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=MODEL_CACHE_DIR,
                token=st.session_state.hf_token
            )
            return load_pipeline(filepath, is_local_file=True)
        except Exception as e:
            return f"❌ 下載失敗: {str(e)}"
    return "❌ 未知的模型選擇"

def handle_civitai_model_download(url):
    """處理 Civitai 模型下載"""
    if not url:
        return "請輸入 Civitai 模型 URL"
    
    try:
        path, fname, msg = download_and_backup(
            url, MODEL_CACHE_DIR, 
            st.session_state.civit_token, 
            st.session_state.hf_token
        )
        result = load_pipeline(path, is_local_file=True)
        return f"{msg}\n{result}"
    except Exception as e:
        return f"❌ 下載失敗: {str(e)}"

# ============== LoRA 處理 ==============
def get_available_loras():
    """取得可用的 LoRA 列表"""
    loras = []
    for folder in [LORA_CACHE_DIR, "./custom_loras"]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith((".safetensors", ".bin")):
                    loras.append(os.path.join(folder, f))
    return loras

def handle_lora_dropdown(lora_path, scale):
    """處理 LoRA 載入"""
    if not lora_path or not os.path.exists(lora_path):
        return "請選擇有效的 LoRA"
    
    if st.session_state.pipe is None:
        return "❌ 請先載入模型"
    
    try:
        st.session_state.pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path),
            cross_attention_scale=scale
        )
        st.session_state.active_loras[lora_path] = scale
        return f"✅ LoRA 載入成功: {os.path.basename(lora_path)}"
    except Exception as e:
        return f"❌ LoRA 載入失敗: {str(e)}"

def handle_lora_download(url, scale):
    """處理 LoRA 下載"""
    if not url:
        return "請輸入 LoRA URL"
    
    if st.session_state.pipe is None:
        return "❌ 請先載入模型"
    
    try:
        path, fname, msg = download_and_backup(
            url, LORA_CACHE_DIR,
            st.session_state.civit_token,
            st.session_state.hf_token
        )
        result = handle_lora_dropdown(path, scale)
        return f"{msg}\n{result}"
    except Exception as e:
        return f"❌ 下載失敗: {str(e)}"

def clear_loras():
    """清除所有 LoRA"""
    if st.session_state.pipe is None:
        return "沒有載入的模型"
    
    try:
        st.session_state.pipe.unload_lora_weights()
        st.session_state.active_loras = {}
        return "✅ 已清除所有 LoRA"
    except Exception as e:
        return f"❌ 清除失敗: {str(e)}"

# ============== 圖片生成 ==============
def generate_image(prompt, neg_prompt, steps, cfg, seed, width, height):
    """生成圖片"""
    if st.session_state.pipe is None:
        return None, "❌ 請先載入模型"

    try:
        # 清理記憶體
        clear_memory()
        
        # 設定 scheduler (使用 DPMSolverMultistepScheduler)
        st.session_state.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            st.session_state.pipe.scheduler.config
        )

        # 生成參數
        generator = torch.Generator("cpu").manual_seed(seed)

        # 執行生成
        with st.spinner("生成中..."):
            result = st.session_state.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator,
            )

        image = result.images[0]

        # 儲存到歷史記錄 (包含完整元資料)
        history_path = save_to_history(
            image, prompt, neg_prompt, seed, steps, cfg,
            width, height, st.session_state.current_model_path
        )

        # 儲存到 outputs 資料夾 (簡單備份)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"image_{timestamp}_seed{seed}.png")
        image.save(save_path)

        # 加入到已生成列表
        st.session_state.generated_images.append({
            "image": image,
            "path": save_path,
            "history_path": history_path,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height
        })

        # 生成後清理記憶體
        clear_memory()

        return image, f"✅ 生成完成! 已儲存至 {save_path}"

    except Exception as e:
        clear_memory()
        return None, f"❌ 生成失敗: {str(e)}"

# ============== 主介面 ==============
def main():
    st.set_page_config(
        page_title="Turbo CPU SD + 永久圖庫",
        page_icon="⚡",
        layout="wide"
    )
    
    # 初始化 session state
    init_session_state()
    
    # 標題
    st.title("⚡ Turbo CPU SD (NSFW + SDXL/Pony 支援)")
    st.markdown("---")
    
    # 側邊欄 - 授權設定
    with st.sidebar:
        st.header("⚙️ 授權金鑰設定")
        st.session_state.hf_token = st.text_input(
            "HF Token", 
            value=st.session_state.hf_token,
            type="password"
        )
        st.session_state.civit_token = st.text_input(
            "Civitai Token", 
            value=st.session_state.civit_token,
            type="password"
        )
        
        st.markdown("---")
        st.header("📊 狀態")
        if st.session_state.current_model_path:
            st.success(f"目前模型: {st.session_state.current_model_path}")
        else:
            st.warning("尚未載入模型")

        if st.session_state.active_loras:
            st.info(f"已載入 LoRA: {len(st.session_state.active_loras)} 個")

        st.markdown("---")
        st.header("🖼️ 最近生成")
        if st.session_state.generated_images:
            for img_data in reversed(st.session_state.generated_images[-3:]):
                st.image(img_data["image"], caption=f"Seed: {img_data['seed']}", use_container_width=True)
        else:
            st.write("尚無生成的圖片")
        
        # 歷史記錄連結
        st.markdown("---")
        st.header("📚 歷史記錄")
        history_count = len(glob.glob(os.path.join(HISTORY_DIR, "*.json")))
        st.write(f"已儲存 {history_count} 張圖片")
    
    # 主要內容區
    col_left, col_right = st.columns([1, 2])
    
    # 左側 - 模型與 LoRA 控制
    with col_left:
        # 模型選擇 Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📦 預設模型", "🔍 HF 搜尋", "📁 HF 檔案模型", "🔗 Civitai 下載"])

        with tab1:
            st.subheader("預設模型")
            preset_choice = st.selectbox(
                "選擇預設模型",
                options=[""] + list(PRESET_MODELS.keys()),
                key="preset_model_select"
            )
            if st.button("載入預設模型", key="load_preset"):
                with st.spinner("載入中..."):
                    st.session_state.status_message = handle_model_dropdown(preset_choice)

        with tab2:
            st.subheader("HuggingFace 模型搜尋")
            hf_search_query = st.text_input("搜尋模型", placeholder="輸入關鍵字搜尋...", key="hf_search_query")
            if st.button("🔍 搜尋", key="search_hf"):
                if hf_search_query:
                    with st.spinner("搜尋中..."):
                        st.session_state.hf_search_results = search_hf_models(hf_search_query)

            if st.session_state.hf_search_results:
                st.write(f"找到 {len(st.session_state.hf_search_results)} 個模型:")
                for model in st.session_state.hf_search_results[:5]:
                    with st.container():
                        st.write(f"**{model['id']}**")
                        st.caption(f"⬇️ {model['downloads']:,} | ❤️ {model['likes']}")
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            if st.button("📥 下載", key=f"dl_{model['id'].replace('/', '_')}"):
                                st.session_state.status_message = download_hf_model(model['id'])
                        with col_m2:
                            if st.button("⚡ 直接載入", key=f"load_{model['id'].replace('/', '_')}"):
                                st.session_state.status_message = load_pipeline(model['id'])
                        with col_m3:
                            # 顯示模型頁面連結
                            st.link_button("🔗 開啟頁面", f"https://huggingface.co/{model['id']}")
                        st.markdown("---")

        with tab3:
            st.subheader("HF 檔案模型")
            hf_choice = st.selectbox(
                "選擇 HF 檔案模型",
                options=[""] + list(HF_FILE_MODELS.keys()),
                key="hf_model_select"
            )
            if st.button("載入 HF 模型", key="load_hf"):
                with st.spinner("載入中..."):
                    st.session_state.status_message = handle_hf_file_model(hf_choice)

        with tab4:
            st.subheader("Civitai 下載")
            civit_url = st.text_input("Civitai 模型 URL", key="civit_url")
            if st.button("下載並載入", key="download_civit"):
                if civit_url:
                    st.session_state.status_message = handle_civitai_model_download(civit_url)
        
        st.markdown("---")
        
        # LoRA 控制
        st.subheader("🎨 LoRA 控制")
        
        lora_tabs1, lora_tabs2 = st.tabs(["本地 LoRA", "下載 LoRA"])
        
        with lora_tabs1:
            available_loras = get_available_loras()
            lora_choice = st.selectbox(
                "選擇本地 LoRA",
                options=[""] + available_loras,
                format_func=lambda x: os.path.basename(x) if x else "",
                key="lora_select"
            )
            lora_scale = st.slider("LoRA 強度", 0.0, 2.0, 1.0, 0.1, key="lora_scale")
            if st.button("載入 LoRA", key="load_lora"):
                st.session_state.status_message = handle_lora_dropdown(lora_choice, lora_scale)
        
        with lora_tabs2:
            lora_url = st.text_input("LoRA URL", key="lora_url")
            lora_dl_scale = st.slider("下載 LoRA 強度", 0.0, 2.0, 1.0, 0.1, key="lora_dl_scale")
            if st.button("下載並載入 LoRA", key="download_lora"):
                st.session_state.status_message = handle_lora_download(lora_url, lora_dl_scale)
        
        if st.button("🗑️ 清除所有 LoRA", key="clear_loras"):
            st.session_state.status_message = clear_loras()
        
        # 顯示狀態訊息
        if st.session_state.status_message:
            st.markdown("---")
            st.info(st.session_state.status_message)
    
    # 右側 - 生成控制
    with col_right:
        st.subheader("🖼️ 圖片生成")
        
        # 提示詞
        prompt = st.text_area(
            "正向提示詞 (Prompt)",
            height=100,
            placeholder="輸入描述您想要生成的圖片內容..."
        )
        
        neg_prompt = st.text_area(
            "負向提示詞 (Negative Prompt)",
            value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            height=80
        )
        
        # 生成參數
        col_param1, col_param2, col_param3 = st.columns(3)
        
        with col_param1:
            steps = st.slider("步數 (Steps)", 1, 50, 20, key="steps")
            cfg = st.slider("CFG Scale", 1.0, 20.0, 7.0, 0.5, key="cfg")
        
        with col_param2:
            width = st.select_slider(
                "寬度",
                options=[384, 512, 640, 768, 896, 1024, 1152, 1280],
                value=512,
                key="width"
            )
            height = st.select_slider(
                "高度",
                options=[384, 512, 640, 768, 896, 1024, 1152, 1280],
                value=512,
                key="height"
            )
        
        with col_param3:
            seed = st.number_input("種子 (Seed)", -1, 999999999, -1, key="seed")
            if seed == -1:
                import random
                seed = random.randint(0, 999999999)

        # 生成按鈕
        if st.button("🎨 生成圖片", type="primary", use_container_width=True):
            if not prompt:
                st.error("請輸入提示詞")
            else:
                image, message = generate_image(
                    prompt, neg_prompt, steps, cfg, seed, width, height
                )
                if image:
                    st.success(message)
                    st.image(image, caption=f"Seed: {seed}", use_container_width=True)
                else:
                    st.error(message)
        
        # 顯示已生成的圖片
        if st.session_state.generated_images:
            st.markdown("---")
            st.subheader("📸 已生成圖片")
            for img_data in reversed(st.session_state.generated_images):
                with st.expander(f"Seed: {img_data['seed']} - {img_data['prompt'][:50]}..."):
                    st.image(img_data["image"], use_container_width=True)
                    st.caption(f"提示詞: {img_data['prompt']}")

if __name__ == "__main__":
    main()
