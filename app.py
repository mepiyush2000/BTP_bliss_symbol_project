from streamlit_image_select import image_select
import streamlit as st
from PIL import Image
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
# for i in range(3):

@st.cache_resource
def get_images():
    l = ["images_with_bliss_id_names" + "/" + i  for i in os.listdir("images_with_bliss_id_names")]
    print("inside image loiader")
    return l[17:50]


@st.cache_resource
def empty_file():
    print("here")
    l=[]
    return l


bliss_symbols = get_images()

print(bliss_symbols)

def read_image(path_x):
    return Image.fromarray(plt.imread(path_x)[:,:,:]).resize(256, 256)

img = image_select(

    label="Select a symbol",
    use_container_width = True,
    images=bliss_symbols,
    # images = [read_image("images_with_bliss_id_names\\bliss_8483.png")]
    # images = [Image.open("images_with_bliss_id_names\\bliss_8483.png")]
    # captions=["A cat", "Another cat", "Oh look, a cat!", "Guess what, a cat..."] + ["cat"]*4,
    # key = str(i)
)
inp = empty_file()

# f = open("cache.txt", "r+")
# if(f.readlines()== []):
#     f.write("cache_line\n")


# f.write(img + "\n")
# f.close()

inp.append(img)
print(" ".join(inp))
st.write(" ".join(inp))

