from configuration.vlm_config import VLMConfig
import torch

def main():
    print("=== NuMarkdown Config Dry Run ===")

    cfg = VLMConfig.from_env()

    print("\n--- Core ---")
    print("Model Path:", cfg.model_path)
    print("Device:", cfg.device)
    print("Torch Dtype:", cfg.torch_dtype)
    print("Device Map:", cfg.device_map)

    print("\n--- Rendering ---")
    print("DPI:", cfg.dpi)
    print("Max Image Long Side:", cfg.max_image_long_side)

    print("\n--- Generation ---")
    print("Max New Tokens:", cfg.max_new_tokens)
    print("Temperature:", cfg.temperature)

    print("\n--- Offline ---")
    print("Transformers Offline:", cfg.transformers_offline)
    print("HF Hub Offline:", cfg.hf_hub_offline)

    print("\n--- Logging ---")
    print("Log Level:", cfg.log_level)

    print("\n--- CUDA Check ---")
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nConfig loaded successfully.")


if __name__ == "__main__":
    main()
