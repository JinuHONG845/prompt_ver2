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
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        return response
    except Exception as e:
        return f"GPT-4 Error: {str(e)}"

def get_claude_response(prompt):
    try:
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        return message
    except Exception as e:
        return f"Claude Error: {str(e)}"

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt, stream=True)
        return response
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def main():
    # CSS로 레이아웃 조정
    st.markdown("""
        <style>
        .stColumn {
            padding: 0 1rem;
        }
        .streamlit-expanderHeader {
            font-size: 1.2em;
        }
        .stMarkdown {
            max-width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("주요 LLM 비교 (v.241111)")
    
    prompt = st.text_area("프롬프트를 입력하세요:", height=100)
    
    if st.button("응답 생성"):
        if prompt:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("GPT-4o")
                response_container = st.empty()
                with st.spinner("GPT-4o 응답 생성 중..."):
                    response = get_gpt4_response(prompt)
                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                full_response += content
                                response_container.markdown(full_response)
            
            with col2:
                st.subheader("Claude-3.5")
                response_container = st.empty()
                with st.spinner("Claude 응답 생성 중..."):
                    response = get_claude_response(prompt)
                    full_response = ""
                    for message in response:
                        if message.type == 'content_block_delta':
                            full_response += message.delta.text
                            response_container.markdown(full_response)
            
            with col3:
                st.subheader("Gemini Pro")
                response_container = st.empty()
                with st.spinner("Gemini Pro응답 생성 중..."):
                    response = get_gemini_response(prompt)
                    full_response = ""
                    for chunk in response:
                        full_response += chunk.text
                        response_container.markdown(full_response)
        else:
            st.warning("프롬프트를 입력해주세요!")

if __name__ == "__main__":
    main() 