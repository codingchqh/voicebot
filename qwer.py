import streamlit as st
from audiorecorder import audiorecorder
print("import 성공")


from gtts import gTTS 
import openai
import os
from datetime import datetime


# Whisper STT
def STT(audio, apikey):
    filename = 'input.mp3'
    audio.export(filename, format='mp3')
    with open(filename, 'rb') as audio_file:6
    client = openai.OpenAI(api_key=apikey)
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    os.remove(filename)
    return transcript.text

# ChatGPT 응답
def ask_gpt(prompt, model, apikey):
    client = openai.OpenAI(api_key=apikey)
    response = client.chat.completions.create(
        model=model,
        messages=prompt
    )
    return response.choices[0].message.content

# TTS 변환
def speak_text(text):
    tts = gTTS(text=text, lang='ko')
    tts.save('output.mp3')
    return 'output.mp3'

# 메인
def main():
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")
    st.header("음성 비서 프로그램")
    st.markdown("---")

    # 기본 설명
    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write("- UI: Streamlit 사용\n- STT: OpenAI Whisper\n- GPT: OpenAI GPT 모델\n- TTS: Google gTTS 사용")

    # session_state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant. Respond in Korean."}]
    if "check_reset" not in st.session_state:
        st.session_state["check_reset"] = False

    # 사이드바
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        model = st.radio("GPT 모델", ["gpt-4", "gpt-3.5-turbo"])
        if st.button("초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant. Respond in Korean."}]
            st.session_state["check_reset"] = True

    # 기능 구현 UI
    col1, col2 = st.columns(2)

    # 질문 녹음
    with col1:
        st.subheader("질문하기")
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중...")
        if audio and audio.duration_seconds > 0 and not st.session_state["check_reset"]:
            st.audio(audio.export().read())
            question = STT(audio, st.session_state["OPENAI_API"])
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"].append(("user", now, question))
            st.session_state["messages"].append({"role": "user", "content": question})

            # GPT 응답
            response = ask_gpt(st.session_state["messages"], model, st.session_state["OPENAI_API"])
            st.session_state["messages"].append({"role": "system", "content": response})
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"].append(("bot", now, response))

            # 음성 재생
            mp3_file = speak_text(response)
            st.audio(mp3_file)

    # 질문/답변 출력
    with col2:
        st.subheader("질문/답변")
        for sender, time, msg in st.session_state["chat"]:
            if sender == "user":
                st.write(f"""
                <div style="display:flex;align-items:center;">
                    <div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{msg}</div>
                    <div style="font-size:0.8rem;color:gray;">{time}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(f"""
                <div style="display:flex;align-items:center;justify-content:flex-end;">
                    <div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{msg}</div>
                    <div style="font-size:0.8rem;color:gray;">{time}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import streamlit as st
from audiorecorder import audiorecorder
print("import 성공")
from gtts import gTTS 
import openai
import os
from datetime import datetime


# Whisper STT
def STT(audio, apikey):
    filename = 'input.mp3'
    audio.export(filename, format='mp3')
    with open(filename, 'rb') as audio_file:6
    client = openai.OpenAI(api_key=apikey)
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    os.remove(filename)
    return transcript.text

# ChatGPT 응답
def ask_gpt(prompt, model, apikey):
    client = openai.OpenAI(api_key=apikey)
    response = client.chat.completions.create(
        model=model,
        messages=prompt
    )
    return response.choices[0].message.content

# TTS 변환
def speak_text(text):
    tts = gTTS(text=text, lang='ko')
    tts.save('output.mp3')
    return 'output.mp3'

# 메인
def main():
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")
    st.header("음성 비서 프로그램")
    st.markdown("---")

    # 기본 설명
    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write("- UI: Streamlit 사용\n- STT: OpenAI Whisper\n- GPT: OpenAI GPT 모델\n- TTS: Google gTTS 사용")


    # session_state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant. Respond in Korean."}]
    if "check_reset" not in st.session_state:
        st.session_state["check_reset"] = False

    # 사이드바
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input("OPENAI API 키", placeholder="Enter Your API Key", type="password")
        model = st.radio("GPT 모델", ["gpt-4", "gpt-3.5-turbo"])
        if st.button("초기화"):
            st.session_state["chat"] = []
            st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant. Respond in Korean."}]
            st.session_state["check_reset"] = True

    # 기능 구현 UI
    col1, col2 = st.columns(2)

    # 질문 녹음
    with col1:
        st.subheader("질문하기")
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중...")
        if audio and audio.duration_seconds > 0 and not st.session_state["check_reset"]:
            st.audio(audio.export().read())
            question = STT(audio, st.session_state["OPENAI_API"])
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"].append(("user", now, question))
            st.session_state["messages"].append({"role": "user", "content": question})

            # GPT 응답
            response = ask_gpt(st.session_state["messages"], model, st.session_state["OPENAI_API"])
            st.session_state["messages"].append({"role": "system", "content": response})
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"].append(("bot", now, response))

            # 음성 재생
            mp3_file = speak_text(response)
            st.audio(mp3_file)

    # 질문/답변 출력
    with col2:
        st.subheader("질문/답변")
        for sender, time, msg in st.session_state["chat"]:
            if sender == "user":
                st.write(f"""
                <div style="display:flex;align-items:center;">
                    <div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{msg}</div>
                    <div style="font-size:0.8rem;color:gray;">{time}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(f"""
                <div style="display:flex;align-items:center;justify-content:flex-end;">
                    <div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{msg}</div>
                    <div style="font-size:0.8rem;color:gray;">{time}</div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
