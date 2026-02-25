import os
import time
import gc
import torch
import requests
import gradio as gr
from huggingface_hub import hf_hub_download, HfApi
from diffusers import (
    AutoPipelineForText2Image, 
    StableDiffusionPipeline,
    StableDiffusionXLPipeline, 
    DPMSolverMultistepScheduler, 
    LCMScheduler,
    EulerDiscreteScheduler 
)

# ── 0. CPU 核心效能最佳化 ──────────────────────────────────────────
torch.set_num_threads(2)

# ── 1. 設定與全域變數 ──────────────────────────────────────────────
MODEL_CACHE_DIR = "./custom_models"
LORA_CACHE_DIR = "./custom_loras"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(LORA_CACHE_DIR, exist_ok=True)

SPACE_ID = os.getenv("SPACE_ID")
ENV_CIVITAI = os.getenv("CIVITAI_TOKEN", "")
ENV_HF = os.getenv("HF_TOKEN", "")

pipe = None
current_model_path = ""
current_model_is_sdxl = False
active_loras = {}  

PRESET_MODELS = {
    "BK-SDM-Tiny (極速輕量 1.5)": "nota-ai/bk-sdm-tiny",
    "Stable Diffusion v1.5 (通用)": "runwayml/stable-diffusion-v1-5",
    "Dreamlike Anime 1.0 (動漫)": "dreamlike-art/dreamlike-anime-1.0",
    "Kernel NSFW (寫實/成人)": "Kernel/sd-nsfw",
    "Realistic Vision V5.1 (高畫質寫實)": "SG161222/Realistic_Vision_V5.1_noVAE",
    "SDXL 1.0 Base (高畫質底模)": "stabilityai/stable-diffusion-xl-base-1.0", 
}

HF_FILE_MODELS = {
    "HomoSimile XL Pony v6 (你的模型 🔑)": ("kines9661/HomoSimile", "homosimileXLPony_v60NAIXLEPSV11.safetensors"),
}

RESOLUTION_CHOICES = [
    384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280
]

def get_model_choices():
    local_models = [f for f in os.listdir(MODEL_CACHE_DIR) if f.endswith(".safetensors")]
    return list(PRESET_MODELS.keys()) + list(HF_FILE_MODELS.keys()) + local_models

def get_lora_choices():
    return [f for f in os.listdir(LORA_CACHE_DIR) if f.endswith(".safetensors")]

# ── 3. 核心邏輯函式 ───────────────────────────────────────────────

def download_and_backup(url, folder, progress, civit_token="", hf_token=""):
    try:
        headers = {}
        if civit_token and civit_token.strip():
            headers["Authorization"] = f"Bearer {civit_token.strip()}"

        progress(0, desc=f"正在連接...")
        response = requests.get(url, stream=True, headers=headers, timeout=15)
        
        if response.status_code in [401, 403]:
            raise Exception("權限不足！請確認 Civitai API Token 是否正確。")
        response.raise_for_status()
        
        fname = "temp_download.safetensors"
        if "content-disposition" in response.headers:
            fname = response.headers["content-disposition"].split("filename=")[-1].strip('"')
        else:
            fname = url.split("/")[-1].split("?")[0]
            if not fname.endswith(".safetensors"): fname += ".safetensors"

        local_filepath = os.path.join(folder, fname)
        
        need_download = True
        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) > 1024 * 1024:
            need_download = False

        if need_download:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024 
            with open(local_filepath, "wb") as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    if total_size > 0:
                        progress(downloaded / total_size, desc=f"下載 {fname[:20]}: {downloaded/1024/1024:.1f}MB")
            
            if os.path.getsize(local_filepath) < 1024 * 100:
                os.remove(local_filepath)
                raise Exception("檔案太小，下載失敗。可能是 NSFW 模型需提供 Token。")

        backup_msg = "✅ (僅暫存)"
        if SPACE_ID and hf_token and hf_token.strip():
            file_size_mb = os.path.getsize(local_filepath) / (1024 * 1024)
            if file_size_mb > 900:
                backup_msg = "⚠️ 檔案>1GB跳過存檔，已暫存。"
                progress(1.0, desc=backup_msg)
            else:
                try:
                    progress(0.9, desc=f"⏳ 正在永久備份到 Space...")
                    api = HfApi(token=hf_token.strip())
                    repo_path = f"{folder.strip('./')}/{fname}"
                    api.upload_file(
                        path_or_fileobj=local_filepath, path_in_repo=repo_path,
                        repo_id=SPACE_ID, repo_type="space"
                    )
                    backup_msg = "✅ 已永久存檔"
                except Exception as upload_err:
                    if "limit reached" in str(upload_err):
                        backup_msg = "⚠️ Space 雲端硬碟已滿，跳過永久存檔。"
                    else:
                        backup_msg = f"⚠️ 存檔失敗，已暫存。"
                    progress(1.0, desc=backup_msg)

        return local_filepath, fname, backup_msg
    except Exception as e:
        raise gr.Error(f"下載失敗: {str(e)}")


def load_pipeline(model_source, is_local_file=False):
    global pipe, current_model_path, current_model_is_sdxl, active_loras
    if model_source == current_model_path and pipe is not None:
        return f"已載入: {os.path.basename(model_source)}"

    pipe = None
    active_loras = {}
    gc.collect()

    # 【修復重點 1】：強制判定是否為 SDXL (從檔名或 Repo 屬性雙重驗證)
    is_sdxl_target = False
    source_lower = model_source.lower()
    if "xl" in source_lower or "pony" in source_lower:
        is_sdxl_target = True

    try:
        if is_local_file:
            if is_sdxl_target:
                p = StableDiffusionXLPipeline.from_single_file(
                    model_source, torch_dtype=torch.float32, 
                    safety_checker=None, requires_safety_checker=False, use_safetensors=True
                )
            else:
                # 若不是 XL 名字，先試 SD1.5，失敗再用 SDXL
                try:
                    p = StableDiffusionPipeline.from_single_file(
                        model_source, torch_dtype=torch.float32, 
                        safety_checker=None, requires_safety_checker=False, use_safetensors=True
                    )
                except Exception:
                    p = StableDiffusionXLPipeline.from_single_file(
                        model_source, torch_dtype=torch.float32, 
                        safety_checker=None, requires_safety_checker=False, use_safetensors=True
                    )
                    is_sdxl_target = True
        else:
            p = AutoPipelineForText2Image.from_pretrained(
                model_source, 
                torch_dtype=torch.float32, 
                safety_checker=None, 
                requires_safety_checker=False
            )

        p.to("cpu")
        p.enable_attention_slicing() 
        
        # 【修復重點 2】：根據最終載入的 Pipeline 類型嚴格判定架構
        if isinstance(p, StableDiffusionXLPipeline) or is_sdxl_target:
            current_model_is_sdxl = True
            model_type_str = "SDXL/Pony XL"
        else:
            current_model_is_sdxl = False
            model_type_str = "SD 1.5"

        pipe = p
        current_model_path = model_source
        return f"✅ 成功載入 ({model_type_str})"
    except Exception as e:
        if is_local_file and os.path.exists(model_source):
            os.remove(model_source) 
        return f"❌ 載入失敗: {str(e)}"

# ── 4. UI 互動事件處理 ─────────────────────────────────────────────

def handle_model_dropdown(choice, hf_token_val):
    if choice in PRESET_MODELS:
        source = PRESET_MODELS[choice]
        yield "⏳ 載入模型中 (若為 SDXL 可能需 2 分鐘，請耐心等待)..."
        yield load_pipeline(source, is_local_file=False)

    elif choice in HF_FILE_MODELS:
        repo_id, filename = HF_FILE_MODELS[choice]
        yield f"⏳ 正在從 HF Hub 下載 {filename}... (首次需時較長)"
        try:
            token = hf_token_val.strip() if hf_token_val and hf_token_val.strip() else None
            local_path = hf_hub_download(
                repo_id=repo_id, 
                filename=filename, 
                token=token,
                local_dir=MODEL_CACHE_DIR
            )
            yield "⏳ 下載完成！正在載入模型..."
            yield load_pipeline(local_path, is_local_file=True)
        except Exception as e:
            yield f"❌ 下載失敗: {str(e)}。若為私人倉庫請確認 HF Token 已填入。"
    else:
        source = os.path.join(MODEL_CACHE_DIR, choice)
        yield "⏳ 載入模型中..."
        yield load_pipeline(source, is_local_file=True)

def handle_civitai_model_download(url, civit_token, hf_token, progress=gr.Progress()):
    if not url: 
        yield "❌ 請輸入網址", gr.update()
        return
    yield "⏳ 下載與處理中...", gr.update()
    try:
        path, fname, backup_msg = download_and_backup(url, MODEL_CACHE_DIR, progress, civit_token, hf_token)
        yield f"⏳ 載入模型中... ({backup_msg})", gr.update()
        status = load_pipeline(path, True)
        choices = get_model_choices()
        yield f"{status} | {backup_msg}", gr.update(choices=choices, value=fname)
    except Exception as e:
        yield f"❌ 錯誤: {e}", gr.update()

def update_lora_list_text():
    if not active_loras: return "無"
    return "\n".join([f"- {k}: {v}" for k, v in active_loras.items()])

def handle_lora_dropdown(lora_filename, scale):
    global pipe, active_loras
    if pipe is None: return "⚠️ 請先載入主模型", update_lora_list_text()
    if not lora_filename: return "⚠️ 未選擇 LoRA", update_lora_list_text()
    path = os.path.join(LORA_CACHE_DIR, lora_filename)
    adapter_name = lora_filename.replace(".", "_")
    try:
        pipe.load_lora_weights(path, adapter_name=adapter_name)
        active_loras[adapter_name] = float(scale)
        return f"✅ 已加入: {lora_filename}", update_lora_list_text()
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg or "No modules were targeted" in error_msg:
            return f"❌ 架構不符！LoRA 與主模型不相容。", update_lora_list_text()
        return f"❌ LoRA 載入失敗: {error_msg}", update_lora_list_text()

def handle_lora_download(url, scale, civit_token, hf_token, progress=gr.Progress()):
    global pipe, active_loras
    if pipe is None: return "⚠️ 請先載入主模型", update_lora_list_text(), gr.update()
    try:
        path, fname, backup_msg = download_and_backup(url, LORA_CACHE_DIR, progress, civit_token, hf_token)
        adapter_name = fname.replace(".", "_")
        try:
            pipe.load_lora_weights(path, adapter_name=adapter_name)
            active_loras[adapter_name] = float(scale)
            choices = get_lora_choices()
            return f"✅ 已套用 {fname} | {backup_msg}", update_lora_list_text(), gr.update(choices=choices, value=fname)
        except Exception as e:
            if adapter_name in active_loras: del active_loras[adapter_name]
            error_msg = str(e)
            if "size mismatch" in error_msg or "No modules were targeted" in error_msg:
                return f"❌ 架構不符！LoRA 與主模型不相容。", update_lora_list_text(), gr.update()
            return f"❌ LoRA 載入失敗: {error_msg}", update_lora_list_text(), gr.update()
    except Exception as e:
        return f"❌ 錯誤: {e}", update_lora_list_text(), gr.update()

def clear_loras():
    global active_loras
    if pipe is None: return "⚠️ 無模型"
    active_loras = {}
    return "🗑️ 已移除所有自訂 LoRA"

# ── 5. 生成圖片 ───────────────────────────────────────────────────

def generate_image(prompt, neg, steps, cfg, seed, width, height, use_lcm):
    if pipe is None: raise gr.Error("請先載入模型！")

    start_time = time.time()
    if seed == -1: seed = int(time.time() % (2**32))
    generator = torch.Generator("cpu").manual_seed(seed)
    
    adapters_to_use = []
    weights_to_use = []
    pipe.unload_lora_weights()
    pipe.disable_lora()
    warning_msg = ""

    # 【修復重點 3】：更精準的加速 LoRA 分配邏輯
    if use_lcm:
        if current_model_is_sdxl:
            # 確認為 SDXL / Pony 模型，掛載 SDXL 專用 Lightning LoRA
            try:
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
                lightning_ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
                pipe.load_lora_weights(lightning_ckpt, adapter_name="lightning")
                adapters_to_use.append("lightning")
                weights_to_use.append(1.0)
                warning_msg = "⚡ SDXL Lightning 已啟動。建議 Steps=4~8, CFG=1.0~2.0。 "
            except Exception as e:
                warning_msg = f"⚠️ Lightning 載入失敗 ({str(e)[:50]})，退回一般模式。 "
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            # 確認為 SD1.5 模型，掛載 SD1.5 專用 LCM LoRA
            try:
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
                adapters_to_use.append("lcm")
                weights_to_use.append(1.0)
                warning_msg = "⚡ LCM 已啟動。建議 Steps=4~8, CFG=1.0~2.0。 "
            except Exception as e:
                warning_msg = f"⚠️ LCM 載入失敗 ({str(e)[:50]})，退回一般模式。 "
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    for k, v in active_loras.items():
        try:
            lora_filename = k.replace("_", ".")
            path = os.path.join(LORA_CACHE_DIR, lora_filename)
            pipe.load_lora_weights(path, adapter_name=k)
            adapters_to_use.append(k)
            weights_to_use.append(v)
        except Exception:
            pass 

    if len(adapters_to_use) > 0:
        pipe.enable_lora()
        pipe.set_adapters(adapters_to_use, adapter_weights=weights_to_use)

    # 生成影像
    image = pipe(
        prompt=prompt, 
        negative_prompt=neg if not use_lcm else None,
        num_inference_steps=int(steps), 
        guidance_scale=float(cfg), 
        width=int(width), height=int(height),
        generator=generator
    ).images[0]
    
    cost_time = time.time() - start_time
    return image, warning_msg + f"✅ 完成 | {width}x{height} | 耗時: {cost_time:.1f}s | Seed: {seed}"


# ── 6. Gradio UI 介面設計 ──────────────────────────────────────────

with gr.Blocks(title="Turbo CPU SD + 永久圖庫") as demo:
    gr.Markdown("# ⚡ Turbo CPU SD (NSFW + SDXL/Pony 支援)")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("⚙️ 授權金鑰設定 (已自動帶入)", open=False):
                civit_token = gr.Textbox(label="Civitai API Token", value=ENV_CIVITAI, placeholder="下載 NSFW 模型用", type="password")
                hf_token = gr.Textbox(label="HF Write Token", value=ENV_HF, placeholder="永久備份 + 私人模型用", type="password")
            
            gr.Markdown("### 1. 主模型管理")
            with gr.Tabs():
                with gr.TabItem("🗂️ 選擇圖庫模型"):
                    model_dropdown = gr.Dropdown(choices=get_model_choices(), value=get_model_choices()[0], label="選擇模型", interactive=True)
                    load_model_btn = gr.Button("載入選擇的模型", variant="primary")
                with gr.TabItem("🌐 下載新模型"):
                    civit_ckpt_url = gr.Textbox(label="Checkpoint 網址", placeholder="輸入 Civitai 直連...")
                    download_model_btn = gr.Button("下載、備份並載入")
            
            model_status = gr.Textbox(label="系統狀態", value="未載入", interactive=False)

            gr.Markdown("### 2. LoRA 管理")
            lora_scale = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="LoRA 權重 (Scale)")
            with gr.Tabs():
                with gr.TabItem("🗂️ 選擇圖庫 LoRA"):
                    lora_dropdown = gr.Dropdown(choices=get_lora_choices(), label="選擇已備份的 LoRA", interactive=True)
                    load_lora_btn = gr.Button("➕ 套用此 LoRA")
                with gr.TabItem("🌐 下載新 LoRA"):
                    lora_url = gr.Textbox(label="LoRA 下載網址", placeholder="輸入 Civitai 直連...")
                    download_lora_btn = gr.Button("➕ 下載、備份並套用")
            
            clear_lora_btn = gr.Button("🗑️ 清空所有已套用的 LoRA")
            lora_status = gr.Textbox(label="目前已套用清單", value="無", lines=2, interactive=False)

        with gr.Column(scale=2):
            use_lcm = gr.Checkbox(label="⚡ 啟用極速模式 (SD1.5→LCM / SDXL→Lightning)", value=True)
            
            gr.Markdown("💡 **Pony XL 使用提示**：Prompt 開頭請加 `score_9, score_8_up, score_7_up,`")
            prompt = gr.Textbox(label="Prompt", value="score_9, score_8_up, score_7_up, a beautiful woman, masterpiece", lines=3)
            neg = gr.Textbox(label="Negative Prompt (極速模式下將忽略)", value="score_1, score_2, score_3, low quality, bad anatomy, worst quality", lines=1)
            
            with gr.Row():
                steps = gr.Slider(1, 30, value=5, step=1, label="Steps (極速模式建議 4~8)")
                cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG (Pony 建議 5~7)")
                seed = gr.Number(-1, label="Seed (-1=隨機)", precision=0)
            
            gr.Markdown("*(SD 1.5 建議 512~768；SDXL/Pony 建議 1024)*")
            with gr.Row():
                width = gr.Dropdown(RESOLUTION_CHOICES, value=1024, label="Width")
                height = gr.Dropdown(RESOLUTION_CHOICES, value=1024, label="Height")
                
            gen_btn = gr.Button("✨ 生成圖片", variant="primary", size="lg")
            gen_status = gr.Textbox(label="生成狀態", interactive=False)
            out_img = gr.Image(label="生成結果", type="pil")

    # ── 7. 綁定按鈕事件 ──
    load_model_btn.click(fn=handle_model_dropdown, inputs=[model_dropdown, hf_token], outputs=[model_status])
    download_model_btn.click(fn=handle_civitai_model_download, inputs=[civit_ckpt_url, civit_token, hf_token], outputs=[model_status, model_dropdown])
    
    load_lora_btn.click(fn=handle_lora_dropdown, inputs=[lora_dropdown, lora_scale], outputs=[model_status, lora_status])
    download_lora_btn.click(fn=handle_lora_download, inputs=[lora_url, lora_scale, civit_token, hf_token], outputs=[model_status, lora_status, lora_dropdown])
    clear_lora_btn.click(fn=clear_loras, outputs=[model_status]).then(fn=update_lora_list_text, outputs=[lora_status])
    
    gen_btn.click(fn=generate_image, inputs=[prompt, neg, steps, cfg, seed, width, height, use_lcm], outputs=[out_img, gen_status])

demo.queue().launch()
