"""
    Encode a real image using the pipeline. 
"""
import PIL
from concept_attention import ConceptAttentionFluxPipeline


pipeline = ConceptAttentionFluxPipeline(
    model_name="flux-schnell",    
    device="cpu"
)

image = PIL.Image.open("dragon_image.png")
concepts = ["dragon", "rock", "sky", "sun", "clouds"]

pipeline_output = pipeline.encode_image(
    image=image,
    concepts=concepts,
    prompt="A fire breathing dragon.",
    width=1024,
    height=1024,
)    # arun - what is the dimensions of this ? 

concept_heatmaps = pipeline_output.concept_heatmaps
#print(f"############# Concept heatmaps: {type(concept_heatmaps)}, tensor shape: {concept_heatmaps[0].shape}#########################")

for concept, concept_heatmap in zip(concepts, concept_heatmaps):
    concept_heatmap_binary = (concept_heatmap > 0.5).float()
    concept_heatmap_binary.save(f"arun: {concept}.png")

