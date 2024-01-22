import streamlit as st

st.title('My Web App')

st.text_input('Message', key ='text')

def print_value():
    st.write(st.session_state['text'])
    st.text(st.session_state['text'])

st.button('Click', on_click=print_value)