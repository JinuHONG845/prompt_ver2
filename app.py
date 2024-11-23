import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import plotly.graph_objects as go
import numpy as np

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

def create_radar_chart(scores, model_name):
    categories = ['정확성', '창의성', '논리성', '완성도', '유용성']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=model_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=True,
        title=f"{model_name} 응답 평가"
    )
    
    return fig

def evaluate_response(response_text):
    # 실제 프로덕션에서는 이 부분을 더 정교한 평가 로직으로 대체해야 합니다
    # 현재는 예시로 랜덤 점수를 생성합니다
    return np.random.uniform(3, 5, 5)

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
            # GPT-4o 응답
            st.subheader("GPT-4o")
            gpt4_container = st.empty()
            gpt4_response = ""
            with st.spinner("GPT-4o 응답 생성 중..."):
                response = get_gpt4_response(prompt)
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            gpt4_response += content
                            gpt4_container.markdown(gpt4_response)
            
            # Claude 응답
            st.subheader("Claude-3.5")
            claude_container = st.empty()
            claude_response = ""
            with st.spinner("Claude 응답 생성 중..."):
                response = get_claude_response(prompt)
                for message in response:
                    if message.type == 'content_block_delta':
                        claude_response += message.delta.text
                        claude_container.markdown(claude_response)
            
            # Gemini 응답
            st.subheader("Gemini Pro")
            gemini_container = st.empty()
            gemini_response = ""
            with st.spinner("Gemini Pro 응답 생성 중..."):
                response = get_gemini_response(prompt)
                for chunk in response:
                    gemini_response += chunk.text
                    gemini_container.markdown(gemini_response)

            # 응답 평가 및 레이더 차트 표시
            st.subheader("응답 평가")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gpt4_scores = evaluate_response(gpt4_response)
                st.plotly_chart(create_radar_chart(gpt4_scores, "GPT-4o"), use_container_width=True)
            
            with col2:
                claude_scores = evaluate_response(claude_response)
                st.plotly_chart(create_radar_chart(claude_scores, "Claude-3.5"), use_container_width=True)
            
            with col3:
                gemini_scores = evaluate_response(gemini_response)
                st.plotly_chart(create_radar_chart(gemini_scores, "Gemini Pro"), use_container_width=True)

            # 평가 기준 설명
            st.subheader("평가 기준 설명")
            st.markdown("""
            - **정확성**: 제공된 정보의 사실적 정확도와 신뢰성
            - **창의성**: 독창적이고 혁신적인 아이디어 제시 능력
            - **논리성**: 논리적 구조와 일관성 있는 설명
            - **완성도**: 응답의 포괄성과 완결성
            - **유용성**: 실제 적용 가능성과 실용적 가치
            """)
        else:
            st.warning("프롬프트를 입력해주세요!")

if __name__ == "__main__":
    main() 