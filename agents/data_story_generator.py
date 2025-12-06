
import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

pipe.enable_model_cpu_offload()

# Placeholder for prompt from data_story_teller
prompt = "Create a clean, modern data-story whiteboard styled titled ‘Top-10 Revenue-Generating Films’. Use a white background, minimalist charts, bold colors, and annotation callouts. Show a key insight section highlighting Film #1 ‘TELEGRAPH VOYAGE’ with a large label ‘$231.73 — 30% above #2’ and an annotation ‘Possible seasonal / holiday boost’. Include a horizontal revenue bar chart showing films ranked 1 to 10 with values from $231.73 to $190.78, visually showing a tight revenue band. Mark the median (film #5) at $204.72 and highlight that only 5 films exceed this value. Add small trend notes near the chart: ‘Steady decline’, ‘Rank 1–3 drop > $10’, ‘Ranks 4–10 drop ~ $4.5 per rank’, and ‘No genre clustering visible’. Add short business implications with simple icons, such as: ‘Boost marketing for top 3’, ‘Ensure inventory availability’, and ‘Monitor post-release revenue curve’. Add actionable step sticky-notes: ‘Compare rentals vs revenue’, ‘Evaluate margin for TELEGRAPH VOYAGE’, and ‘Upsell ranks 4–10’. Use flat infographic design, light color palette, simple movie and chart icons, clean sans-serif typography, and a professional business-focused tone."

# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=10,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

# Placeholder for hashed image to be saved in exports folder hashed and returned to user in Slack
image.save("example2-z-image.png")
