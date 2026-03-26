import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/tmp/calme-3.2-instruct-78b-merged"

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model on CPU (this can take a while)...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=True,
)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

prompt = "Hello, my name is"
inputs = tok(prompt, return_tensors="pt")

print("Running generation...", flush=True)
out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
text = tok.decode(out[0], skip_special_tokens=True)
print("GENERATION_OK")
print(text)
