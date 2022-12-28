import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import base64
from datetime import datetime

st.set_page_config(layout="wide")

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

letters = ['ㅂ','ㅅ','ㅅ','ㅁ','챗봇']
text=""
text += f'<span style="color:red">{letters[0]}<span>'
text += f'<span style="color:blue">{letters[1]}<span>'
text += f'<span style="color:#ffd400">{letters[2]}<span>'
text += f'<span style="color:green">{letters[3]}<span>'
text += f'<span style="color:#ff6633"> {letters[4]}<span>' #three-per-em space " "

st.markdown(f"<h2 style='text-align: center;'>{text}</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:black;'>(ง˙∇˙)ว 안녕하세요! 부산소마고 챗봇입니다 (ง˙∇˙)ว</h3>", unsafe_allow_html=True) 

st.write('') #개행

#사용자 채팅 입력폼
with st.form('form',clear_on_submit=True):
    st.markdown("""<p class="user-explain">학교에 대한 질문을 해보세요</p>""", unsafe_allow_html=True)
    user_input = st.text_input('')
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
    if answer['distance'] > 0.6: #질문이 학습자료와 유사도가 60%이상이라면 
        st.session_state.generated.append(answer['챗봇']) #대답
    else: #아니면
        st.session_state.generated.append("잘 모르겠습니다. 학교 연락처로 질문해보세요 (051-971-2153)") #예외처리


#css 불러오기
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

#이미지 불러오기
file = open("./image/아리.png","rb")
contents = file.read()
data_url = base64.b64encode(contents).decode("UTF-8")

file2 = open("./image/소리.png","rb")
contents2 = file2.read()
data_url2 = base64.b64encode(contents2).decode("UTF-8")

#채팅 내역 표시 및 채팅 UI 구현하기 
now = datetime.now()
current_time = now.strftime("%H:%M")
print(current_time)

for i in range(len(st.session_state['past'])):
    st.markdown(
        """
        <div class="rigth-msg">
            <div class="right-msg-form">
                <div class="right-msg-time">{4}</div>
                <p class="right-msg-p">{0}</p>
            </div>
            <div>
                <img src="data:image/png;base64,{2}" width="50" height="60"/>
            </div>
        </div>
        <div class="left-msg">
        <div class="left-img">
            <div class="left-img-div">
                <img src="data:image/png;base64,{3}" width="50" height="70"/>
            </div>
        </div>
            <div class="left-msg-form">
                <div class="left-msg-time">{4}</div>
                <p class="left-msg-p">{1}</p>
            </div>
        </div>
        """.format(st.session_state['past'][i], st.session_state['generated'][i], data_url, data_url2, current_time),unsafe_allow_html=True)

#사이드바
st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage](https://school.busanedu.net/bssm-h/main.do) |
    [Instagram](https://www.instagram.com/bssm.hs/) |
    [Facebook](https://www.facebook.com/BusanSoftwareMeisterHighschool)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    📞 051-971-2153
    """
)
