from huggingface_hub import HfFileSystem

token = ""
fs = HfFileSystem(token=token)

print(fs.ls("CemigP/Rag_Cemig", detail=False))

# print(fs.rm("CemigP/Rag_Cemig/e5_large_ft_v2/model.safetensors"))
# print(fs.rm("CemigP/Rag_Cemig/reranker_ft/model.safetensors"))

# Commands to upload the models to Hugging Face Hub:
# hf upload CemigP/Rag_Cemig ./experiments/exp_v2_40k/models/e5_large_ft_v2/model.safetensors e5_large_ft_v2/model.safetensors
# hf upload CemigP/Rag_Cemig ./experiments/exp_v2_40k_reranker_ft/models/reranker_ft/model.safetensors reranker_ft/model.safetensors
