import base64
import pkgutil
import streamlit as st
from st_clickable_images import clickable_images
import os
from prediction_file import *

@st.cache_resource
def get_images():
    f = open("popular_bliss.txt", "r")
    files = list(map(lambda x:x.strip("\n"), f.readlines()))

    l = ["images_with_bliss_id_names" + "/" + i  for i in files]
    caption = files
    l.sort(), caption.sort()
    print("inside image loiader")
    print(caption)
    return l[17:50], caption[17:50]


@st.cache_resource
def empty_file():
    print("here")
    l=[]
    l2 = []
    return l, l2

@st.cache_resource
def load_model(path = "models"):
    inp_vectorization_, out_vectorization_, transformer = load_model_files(path)
    return inp_vectorization_, out_vectorization_, transformer



inp_vectorization_, out_vectorization_, transformer = load_model("models")

bliss_symbols, title = get_images()
# print(title)


st.write("Select from symbols: ")

images = []
for file in  bliss_symbols:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")

clicked = clickable_images(
    images,
    # titles= [str(i) for i in range(17, 100)],
    titles = [i.split("__")[1][:-4] for i in title],
    div_style={"display": "flex", "justify-content": "space-between", "flex-wrap": "wrap"},
    img_style={"margin": "12px", "height": "85px", "border": "2.5px solid #BEBEBE", "border-radius":"10px", "padding": "10px", "cursor":"pointer"},
    
)

st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")


inp, wrt = empty_file()
inp.append(title[clicked].split("_")[1])
wrt.append(title[clicked].split("__")[1][:-4])

print(inp)
st.write(" | ".join(wrt[1:]))

seq_input = " ".join(inp[1:-1])

if st.button('Run'):
    print(seq_input)
    res = decode_sequence(seq_input, inp_vectorization_, out_vectorization_, transformer)
    st.title(" ".join(res.split(" ")[1:-1]))
    # st.markdown(decode_sequence('14916 24732 18465 14449', inp_vectorization_, out_vectorization_, transformer))
else:
    pass




option = st.selectbox(
    'Convert to Tense',
    ('PAST', 'PRESENT', 'FUTURE'))


if (option=="PAST"):
    res = decode_sequence(seq_input, inp_vectorization_, out_vectorization_, transformer)
    text = " ".join(res.split(" ")[1:-1])
    changed_tense_sent = change_tense(text, "past")
elif (option=="PRESENT"):
    res = decode_sequence(seq_input, inp_vectorization_, out_vectorization_, transformer)
    text = " ".join(res.split(" ")[1:-1])
    changed_tense_sent = change_tense(text, "present")
elif (option=="FUTURE"):
    res = decode_sequence(seq_input, inp_vectorization_, out_vectorization_, transformer)
    text = " ".join(res.split(" ")[1:-1])
    changed_tense_sent = change_tense(text, "future")

st.title(changed_tense_sent)

# else:
#     st.write('Goodbye')