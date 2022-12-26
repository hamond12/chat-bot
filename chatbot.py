import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model=SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('chatbot.csv')
    df['embedding']=df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.title('(ง˙∇˙)ว 부소마 챗봇 (ง˙∇˙)ว')
st.subheader("안녕하세요! 소마고 챗봇입니다.")
st.subheader("학교에 대해 궁금한 것을 물어보세요.")

#사용자 채팅 입력폼
with st.form('form',clear_on_submit=True):
    user_input = st.text_input('사용자 : ','')
    submitted = st.form_submit_button('전송')

#응답
if 'generated' not in st.session_state:
    st.session_state['generated']=[]

#질문
if 'past' not in st.session_state:
    st.session_state['past']=[]

if submitted and user_input:
    embedding = model.encode(user_input)
    
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()] #idxmax: unknown word

    st.session_state.past.append(user_input)
    if answer['distance'] > 0.5:
        st.session_state.generated.append(answer['챗봇'])
    else:
        st.session_state.generated.append("뭐라노")

#채팅 내역 표시
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')
    if len(st.session_state['generated'])>i:
        message(st.session_state['generated'][i], key=str(i)+'_bot')