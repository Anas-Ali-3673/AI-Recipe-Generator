# Recipe Generator Streamlit App

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run the App
```bash
streamlit run recipe_app.py
```

### 3. Use the App
- Enter a recipe name
- Select genre from dropdown
- Optionally add ingredients
- Adjust creativity slider
- Click "Generate Recipe"

## Features
- ✅ Real-time recipe generation
- ✅ Adjustable parameters (length, temperature)
- ✅ Multiple genre support
- ✅ Download generated recipes
- ✅ Clean, intuitive UI

## Note
This uses the base GPT-2 model. For better results:
1. Train the model using the notebook
2. Save the fine-tuned model
3. Update the model path in `recipe_app.py`

## Model Path Update (After Training)
Replace line 11-12 in `recipe_app.py`:
```python
# Change from:
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# To:
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
```
