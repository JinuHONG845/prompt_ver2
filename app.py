import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# API 키 설정
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_gpt4_response(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT-4 Error: {str(e)}"

def get_claude_response(prompt):
    try:
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude Error: {str(e)}"

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def main():
    st.title("LLM 응답 비교기")
    
    # 메인 입력 영역
    prompt = st.text_area("프롬프트를 입력하세요:", height=100)
    
    if st.button("응답 생성"):
        if prompt:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("GPT-4 응답")
                with st.spinner("GPT-4 응답 생성 중..."):
                    gpt4_response = get_gpt4_response(prompt)
                    st.write(gpt4_response)
            
            with col2:
                st.subheader("Claude-3 응답")
                with st.spinner("Claude 응답 생성 중..."):
                    claude_response = get_claude_response(prompt)
                    st.write(claude_response)
            
            with col3:
                st.subheader("Gemini Pro 응답")
                with st.spinner("Gemini 응답 생성 중..."):
                    gemini_response = get_gemini_response(prompt)
                    st.write(gemini_response)
        else:
            st.warning("프롬프트를 입력해주세요!")

if __name__ == "__main__":
    main() 