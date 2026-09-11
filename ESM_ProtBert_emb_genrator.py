#ESM  & Prot_bert

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer

# =====================
# CONFIGURATION
# =====================
DEFAULT_BATCH_SIZE = 32
MAX_LEN = 1024
OUTPUT_DIR = "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ProteinEmbedder:
    def __init__(self, model_name, output_name, space_separated=False, use_t5_encoder=False):
        """
        Args:
            model_name (str): HuggingFace model path.
            output_name (str): Filename for saving embeddings.
            space_separated (bool): If True, adds spaces between amino acids (e.g., "M A K ...").
            use_t5_encoder (bool): If True, uses T5EncoderModel (specific for T5-based models).
        """
        self.model_name = model_name
        self.output_name = output_name
        self.space_separated = space_separated
        self.use_t5_encoder = use_t5_encoder
        self.device = DEVICE
        self.model = None
        self.tokenizer = None

    def load_model(self):
        print(f"\n🔹 Loading {self.model_name} on {self.device}...")
        try:
            if self.use_t5_encoder:
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
                self.model = T5EncoderModel.from_pretrained(self.model_name)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
                self.model = AutoModel.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Enable half precision for faster inference on CUDA
            if self.device == "cuda":
                self.model.half()
                
        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {e}")
            self.model = None

    def preprocess_sequences(self, seqs):
        """Applies space separation if needed."""
        if self.space_separated:
            return [" ".join(list(seq)) for seq in seqs]
        return seqs

    def get_embeddings(self, seqs, batch_size=DEFAULT_BATCH_SIZE):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return None

        processed_seqs = self.preprocess_sequences(seqs)
        all_embeddings = []
        
        # Sort sequences by length for efficient batching (optional, but recommended)
        # We will keep track of original indices to restore order later
        original_indices = np.argsort([len(s) for s in processed_seqs])[::-1]
        sorted_seqs = [processed_seqs[i] for i in original_indices]

        print(f"🚀 Generating embeddings for {len(seqs)} sequences (Batch size: {batch_size})...")

        with torch.inference_mode(): # More efficient than no_grad
            for i in tqdm(range(0, len(sorted_seqs), batch_size)):
                batch_seqs = sorted_seqs[i : i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_seqs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LEN
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Run Model
                outputs = self.model(**inputs)

                # Extract Embeddings
                # Strategy: Mean pooling over non-padded tokens
                if hasattr(outputs, "last_hidden_state"):
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    
                    # Mask padding tokens
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    # Fallback if no last_hidden_state (unlikely for these models)
                    batch_embeddings = outputs[0].mean(dim=1).cpu().numpy()
                
                all_embeddings.append(batch_embeddings)

        # Concatenate and restore original order
        all_embeddings = np.vstack(all_embeddings)
        
        # Restore order
        restored_embeddings = np.zeros_like(all_embeddings)
        for i, original_idx in enumerate(original_indices):
            restored_embeddings[original_idx] = all_embeddings[i]

        return restored_embeddings

    def save_embeddings(self, embeddings):
        if embeddings is not None:
            save_path = os.path.join(OUTPUT_DIR, f"{self.output_name}.npy")
            np.save(save_path, embeddings)
            print(f"✅ Saved: {save_path} → shape {embeddings.shape}")
            
            # Display first 10 embeddings to confirm they are different
            print(f"\n📊 First 10 embeddings preview for {self.output_name}:")
            print(embeddings[:10])
            print("=" * 50 + "\n")

    def run(self, seqs):
        embeddings = self.get_embeddings(seqs)
        self.save_embeddings(embeddings)
        # Clear memory
        self.model = None
        self.tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

# =====================
# PIPELINE DEFINITION
# =====================
def run_pipeline(file_path):
    # 1. Load Data
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Loading data from {file_path}...")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        print("❌ Unsupported file format. Use CSV or XLSX.")
        return

    if "sequence" not in df.columns:
        print("❌ Column 'sequence' not found in file.")
        return

    seqs = df["sequence"].astype(str).tolist()
    print(f"✅ Loaded {len(seqs)} sequences.")

    # 2. Define Models
    models_to_run = [
        # ESM-2 (Facebook/Meta) - No spaces
        ProteinEmbedder(
            model_name="facebook/esm2_t33_650M_UR50D", 
            output_name="emb_train_esm", 
            space_separated=False
        ),
        # ProtBERT (Rostlab) - Requires spaces, BERT-based
        ProteinEmbedder(
            model_name="Rostlab/prot_bert", 
            output_name="emb_train_protbert", 
            space_separated=True
        ),
        
    ]

    # Note: facebook/esm3 is not currently available on HuggingFace Hub in the same format.
    # If you have access to it via a private repo or API, add it here.

    # 3. Execution
    for embedder in models_to_run:
        embedder.run(seqs)

    print("\n🎉 All tasks completed!")

if __name__ == "__main__":
    # Static file path configuration
    STATIC_FILE_PATH = "/kaggle/input/datasets/engrsiyab/try-001/train.xlsx"
    run_pipeline(STATIC_FILE_PATH)
