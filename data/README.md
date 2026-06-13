# Relay ML Data

Track every dataset used by the ML service here.

## Required First Downloads

| Dataset | Source | Use | Status |
|---|---|---|---|
| AI-generated e-commerce defect images | HuggingFace: `prajwalkothwal/ai-generated-ecommerce-images` | Grade CNN, defect labels, demo images | pending |
| Clothing fit dataset | Kaggle: `rmisra/clothing-fit-dataset-for-size-recommendation` | Fit flags and size signals | pending |

## Documentation Template

For each dataset, add:

- source URL
- license or terms
- downloaded date
- local path
- row/image count
- labels used
- train/validation/test split
- any preprocessing steps

Do not commit large raw datasets unless the repo is configured for Git LFS or
the team explicitly agrees to store a small demo subset.
