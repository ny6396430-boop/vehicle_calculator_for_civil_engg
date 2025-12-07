import streamlit as st
import subprocess
import tempfile
import os
import shutil

st.set_page_config(page_title='Vehicle Calculator for Civil Engg', layout='wide')
st.title('Vehicle Calculator for Civil Engg')

uploaded = st.file_uploader('Upload recorded video (mp4, avi, mov)', type=['mp4','avi','mov'])

if uploaded is not None:
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, 'wb') as f:
        f.write(uploaded.getbuffer())

    st.video(input_path)

    st.markdown('**Detection settings**')
    conf = st.slider('Confidence threshold', 0.1, 0.9, 0.4)
    model_name = st.selectbox('YOLOv8 model', ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'])

    if st.button('Run detection & count'):
        output_path = os.path.join(tmp_dir, 'annotated_' + uploaded.name)
        cmd = [
            'python', 'detect_and_count.py',
            '--source', input_path,
            '--output', output_path,
            '--model', model_name,
            '--conf', str(conf)
        ]
        with st.spinner('Running detection (this may take a while)...'):
            proc = subprocess.run(cmd, capture_output=True, text=True)

        st.text('--- STDOUT ---')
        st.text(proc.stdout)
        st.text('--- STDERR (if any) ---')
        st.text(proc.stderr)

        if os.path.exists(output_path):
            st.video(output_path)
            st.success('Done. Download below:')
            with open(output_path, 'rb') as f:
                st.download_button('Download annotated video', f, file_name=os.path.basename(output_path))
        else:
            st.error('Output not found. Check logs above.')

    st.info('Temporary files are stored while the app runs; they will be cleaned when the session ends.')
