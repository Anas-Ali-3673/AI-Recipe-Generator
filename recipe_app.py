import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

st.set_page_config(page_title="Recipe Generator", page_icon="🍳", layout="wide")

# Cache model loading
@st.cache_resource
def load_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_recipe(prompt, tokenizer, model, max_length=200, temperature=0.8):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.title("🍳 AI Recipe Generator")
st.markdown("Generate recipes using GPT-2 Language Model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    genre = st.selectbox("Recipe Genre", 
                         ["vegetables", "drinks", "cereal", "sides", "nonveg", 
                          "fastfood", "bakery", "fusion", "meal"])
    
    max_length = st.slider("Max Length", 50, 500, 200)
    temperature = st.slider("Creativity", 0.1, 1.5, 0.8)
    
    # st.markdown("---")
    # st.markdown("### 📊 Model Info")
    # st.info("Using GPT-2 base model\n\n**Note:** For fine-tuned model, train using the notebook first")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    
    recipe_name = st.text_input("Recipe Name", placeholder="e.g., Chocolate Cake")
    ingredients = st.text_area("Ingredients (optional)", 
                                placeholder="flour, sugar, eggs, chocolate")
    
    if st.button("🎯 Generate Recipe", type="primary"):
        if recipe_name:
            with st.spinner("Generating recipe..."):
                tokenizer, model = load_model()
                
                if tokenizer and model:
                    # Create prompt
                    prompt = f"Recipe: {recipe_name}\nGenre: {genre}\n"
                    if ingredients:
                        prompt += f"Ingredients: {ingredients}\n"
                    prompt += "Directions:\n"
                    
                    # Generate
                    result = generate_recipe(prompt, tokenizer, model, max_length, temperature)
                    
                    # Store in session state
                    st.session_state['generated_recipe'] = result
                    st.session_state['recipe_name'] = recipe_name
        else:
            st.warning("Please enter a recipe name")

with col2:
    st.subheader("✨ Generated Recipe")
    
    if 'generated_recipe' in st.session_state:
        st.markdown(f"### {st.session_state['recipe_name']}")
        st.markdown(st.session_state['generated_recipe'])
        
        # Download button
        st.download_button(
            label="📥 Download Recipe",
            data=st.session_state['generated_recipe'],
            file_name=f"{st.session_state['recipe_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )
    else:
        st.info("👈 Enter recipe details and click Generate")

# Footer
st.markdown("---")
st.markdown("### 🔑 Key Concepts Used")
cols = st.columns(4)
with cols[0]:
    st.metric("Model", "GPT-2")
with cols[1]:
    st.metric("Technique", "LoRA")
with cols[2]:
    st.metric("Quantization", "4-bit")
with cols[3]:
    st.metric("Library", "Transformers")
