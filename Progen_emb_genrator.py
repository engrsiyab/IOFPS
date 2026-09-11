# ProGen_Embeddings Implementation

import os
import sys
import torch
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# CONFIGURATION
# =====================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_BATCH_SIZE = 4  # Kept small for ProGen2 memory constraints
MAX_LEN = 1024
OUTPUT_DIR = "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ProteinEmbedder:
    def __init__(self, model_name, output_name):
        """
        Args:
            model_name (str): HuggingFace model path.
            output_name (str): Filename for saving embeddings.
        """
        self.model_name = model_name
        self.output_name = output_name
        self.device = DEVICE
        self.model = None
        self.tokenizer = None

    def load_model(self):
        print(f"\n🔹 Loading ProGen ({self.model_name}) on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Causal LMs require left-padding when extracting sequence-level representations
            self.tokenizer.padding_side = "left"
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                output_hidden_states=True,
                trust_remote_code=True
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Half-precision to prevent CUDA OOM on Kaggle
            if self.device == "cuda":
                self.model.half()
                
        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {e}")
            self.model = None

    def get_embeddings(self, seqs, batch_size=DEFAULT_BATCH_SIZE):
        if self.model is None:
            self.load_model()
            if self.model is None:
                print("❌ Aborting task: Model failed to load.")
                return None

        processed_seqs = [str(s).upper() for s in seqs]
        
        # Sort sequences by length for optimal padding efficiency
        original_indices = np.argsort([len(s) for s in processed_seqs])[::-1]
        sorted_seqs = [processed_seqs[i] for i in original_indices]

        all_embeddings = []
        print(f"🚀 Generating ProGen embeddings for {len(seqs)} sequences...")

        with torch.inference_mode():
            for i in tqdm(range(0, len(sorted_seqs), batch_size)):
                batch_seqs = sorted_seqs[i : i + batch_size]
                
                inputs = self.tokenizer(
                    batch_seqs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LEN
                ).to(self.device)

                outputs = self.model(**inputs)

                # Extract last hidden state and compute mean pooling over sequence tokens
                last_hidden_state = outputs.hidden_states[-1] 
                attention_mask = inputs["attention_mask"]
                
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                
                batch_embeddings = (sum_embeddings / sum_mask).detach().cpu().numpy()
                all_embeddings.append(batch_embeddings.astype(np.float32))

                # GPU Memory cleanup
                del inputs, outputs, last_hidden_state, mask_expanded, sum_embeddings, sum_mask
                if i % (batch_size * 5) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # Re-sort embeddings back to match original sequence input order
        all_embeddings = np.vstack(all_embeddings)
        restored_embeddings = np.zeros_like(all_embeddings)
        for i, original_idx in enumerate(original_indices):
            restored_embeddings[original_idx] = all_embeddings[i]

        return restored_embeddings

    def save_embeddings(self, embeddings):
        if embeddings is not None:
            save_path = os.path.join(OUTPUT_DIR, f"{self.output_name}.npy")
            np.save(save_path, embeddings)
            print(f"✅ Saved: {save_path} → shape {embeddings.shape}")

    def run(self, seqs):
        embeddings = self.get_embeddings(seqs)
        self.save_embeddings(embeddings)
        self.model = None  # Free GPU VRAM upon completion
        torch.cuda.empty_cache()

# =====================
# PIPELINE DEFINITION
# =====================
def run_pipeline(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found at path: {file_path}")
        return
    
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    seqs = df["sequence"].astype(str).tolist()

    # Active ProGen2-small Hugging Face repository
    embedder = ProteinEmbedder(
        model_name="hugohrban/progen2-small", 
        output_name="progen_embeddings_test"
    )
    embedder.run(seqs)

    print("\n🎉 ProGen Embedding Task Completed!")

if __name__ == "__main__":
    STATIC_FILE_PATH = "/kaggle/input/datasets/engrsiyab/phd-dataset/test.xlsx"
    run_pipeline(STATIC_FILE_PATH)