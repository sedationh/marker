import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

import base64
import io
import re
import tempfile
from typing import List, Any, Dict

import pypdfium2
import streamlit as st

from marker.convert import convert_single_pdf
from marker.models import load_all_models
from surya.languages import CODE_TO_LANGUAGE

@st.cache_resource()
def load_models():
    return load_all_models()


def convert_pdf(fname: str, langs: List[str] | None, max_pages: int | None, ocr_all_pages: bool, start_page: int | None) -> (str, Dict[str, Any], dict):
    full_text, images, out_meta = convert_single_pdf(fname, model_lst, max_pages=max_pages, langs=langs, ocr_all_pages=ocr_all_pages, start_page=start_page)
    return full_text, images, out_meta


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f'<img src="data:image/png;base64,{encoded}" alt="{img_alt}" style="max-width: 100%;">'
    return img_html


def markdown_insert_images(markdown, images):
    image_tags = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)

    for image in image_tags:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if image_path in images:
            markdown = markdown.replace(image_markdown, img_to_html(images[image_path], image_alt))
    return markdown


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)

# Add this new function to handle Marker execution
def run_marker_process(in_file, languages, max_pages, ocr_all_pages, start_page):
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(in_file.getvalue())
        temp_pdf.seek(0)
        filename = temp_pdf.name
        print(f"start_page: {start_page}")
        md_text, images, out_meta = convert_pdf(filename, languages, max_pages, ocr_all_pages, start_page)
    return markdown_insert_images(md_text, images)

st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])

model_lst = load_models()


st.markdown("""
# Marker Demo

This app will let you try marker, a PDF -> Markdown converter. It works with any languages, and extracts images, tables, equations, etc.

Find the project [here](https://github.com/VikParuchuri/marker).
""")

in_file = st.sidebar.file_uploader("PDF file:", type=["pdf"])
languages = st.sidebar.multiselect("Languages", sorted(list(CODE_TO_LANGUAGE.values())), default=[], max_selections=4, help="Select the languages in the pdf (if known) to improve OCR accuracy.  Optional.")
max_pages = st.sidebar.number_input("Max pages to parse", min_value=1, value=1, help="Optional maximum number of pages to convert")
ocr_all_pages = st.sidebar.checkbox("Force OCR on all pages", help="Force OCR on all pages, even if they are images", value=False)
convert_from_current_page = st.sidebar.checkbox("Convert from current page", help="Start conversion from the currently selected page", value=True)

if in_file is None:
    st.stop()

filetype = in_file.type

with col1:
    page_count = page_count(in_file)
    page_number = st.number_input(f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count, key="page_number")
    pil_image = get_page_image(in_file, page_number)

    st.image(pil_image, caption="PDF file (preview)", use_column_width=True)

# Replace the manual trigger with an automatic one based on page number changes
if 'last_page_number' not in st.session_state:
    st.session_state.last_page_number = page_number

if page_number != st.session_state.last_page_number:
    st.session_state.last_page_number = page_number
    start_page = page_number - 1 if convert_from_current_page else None
    md_text = run_marker_process(in_file, languages, max_pages, ocr_all_pages, start_page)

    with col2:
        st.markdown(f'<div id="markdown_container" contenteditable="true">{md_text}</div>', unsafe_allow_html=True)
