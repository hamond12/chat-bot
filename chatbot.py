import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt #그래프 그리기
from matplotlib import font_manager, rc #한글 폰트 사용
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

#css 불러오기
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")

#로고 색입히기
letters = ['ㅂ','ㅅ','ㅅ','ㅁ','챗봇']
text=""
text += f'<span style="color:red">{letters[0]}<span>'
text += f'<span style="color:blue">{letters[1]}<span>'
text += f'<span style="color:#ffd400">{letters[2]}<span>'
text += f'<span style="color:green">{letters[3]}<span>'
text += f'<span style="color:#ff6633"> {letters[4]}<span>' #three-per-em space " "

#인삿말
st.markdown(f"<h2 style='text-align: center;'>{text}</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color:black;'>(ง˙∇˙)ว 안녕하세요! 부산소마고 챗봇입니다 (ง˙∇˙)ว</h3>", unsafe_allow_html=True) 

#개행
st.write('') 
st.write('') 

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

#탭 추가
tab1, tab2 = st.tabs(["학교소개", "챗봇"])

#학교소개 탭
with tab1:

    st.markdown("""<p class="intro-school">🏫 저희 부산SW마고를 소개합니다.</p>""", unsafe_allow_html=True)

    #학교 사진 불러오기
    file3 = open("./image/school.png","rb")
    contents3 = file3.read()
    data_url3 = base64.b64encode(contents3).decode("UTF-8")
    
    st.markdown("""<img src="data:image/png;base64,{0}" width="1000"/>""".format(data_url3), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="explain-box">
            <br>
            <p class="explain-school">부산광역시 강서구 가락대로 1393에 위치한 남녀공학 특수목적 고등학교입니다.</p>
            <p class="explain-school">1970년 개교하였으며, 2021년 마이스터고로 전환되었습니다. 전국단위로 학생을 모집합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        #학생 수 원형 차트
        font_path = "C:/Windows/Fonts/NanumGothic.ttf"
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)

        labels = ["소프트웨어", "임베디드", "1학년"]
        sizes = [32,32,64]
        group_colors=['lightskyblue','lightcoral','yellowgreen']
        wedgeprops = {
        'width': 0.8,
        'edgecolor': 'w',
        'linestyle': '-',
        'linewidth': 5
        }

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=group_colors,
                startangle=80, 
                wedgeprops=wedgeprops,
                textprops={'fontsize':10})
        ax1.axis('equal') 

        plt.title('학생', size=25)
        plt.figure(figsize=(9,9))
        st.pyplot(fig1)

    with col2:
        font_path = "C:/Windows/Fonts/NanumGothic.ttf"
        font = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font)

        labels = ["보통교사", "전문교과", "비교과교사"]
        sizes = [12,20,4]
        group_colors=['#F1D3B3','#C7BCA1','#8B7E74']
        wedgeprops = {
        'width': 0.8,
        'edgecolor': 'w',
        'linestyle': '-',
        'linewidth': 5
        }

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=group_colors,
                startangle=80, 
                wedgeprops=wedgeprops,
                textprops={'fontsize':10})
        ax1.axis('equal') 

        plt.title('교원', size=25)
        plt.figure(figsize=(12,12))
        st.pyplot(fig1)

#챗봇 탭
with tab2:
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
