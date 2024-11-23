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

# 전역 변수로 categories 정의
categories = ['정확성', '창의성', '논리성', '완성도', '유용성']

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
    # 각 모델별 색상 정의
    colors = {
        'GPT-4o': '#FF6B6B',  # 선명한 빨간색
        'Claude-3.5': '#4ECDC4',  # 청록색
        'Gemini Pro': '#45B7D1'  # 하늘색
    }

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=model_name,
        line=dict(color=colors[model_name]),  # 선 색상
        fillcolor=colors[model_name] + '50'  # 채우기 색상 (투명도 50% 추가)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title=f"{model_name} 응답 평가"
    )
    
    return fig

def evaluate_response(response_text):
    # 실제 프로덕션에서는 이 부분을 더 정교한 평가 로직으로 대체해야 합니다
    # 현재는 예시로 랜덤 점수를 생성합니다
    return np.random.uniform(6, 10, 5)

def get_evaluation_prompt(responses):
    return f"""다음은 동일한 질문에 대한 세 AI 모델의 응답입니다. 각 응답을 객관적으로 평가해주세요.

질문: {responses['prompt']}

GPT-4o의 응답:
{responses['gpt4']}

Claude-3.5의 응답:
{responses['claude']}

Gemini Pro의 응답:
{responses['gemini']}

세 모델의 응답을 비교 분석하여 3줄로 간단히 총평해주세요. 각 모델의 장단점을 객관적으로 평가해주세요."""

def get_summary_evaluation(model_name, responses):
    evaluation_prompt = get_evaluation_prompt(responses)
    
    if model_name == "GPT-4o":
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 AI 응답을 분석하고 평가하는 전문가입니다. 객관적이고 공정한 평가를 제공해주세요."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

    elif model_name == "Claude-3.5":
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            temperature=0.7,
            system="당신은 AI 응답을 분석하고 평가하는 전문가입니다. 객관적이고 공정한 평가를 제공해주세요.",
            messages=[{"role": "user", "content": evaluation_prompt}]
        )
        return response.content[0].text

    else:  # Gemini Pro
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            contents=f"""당신은 AI 응답을 분석하고 평가하는 전문가입니다. 객관적이고 공정한 평가를 제공해주세요.

{evaluation_prompt}""",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )
        return response.text

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

            # 구분선 추가
            st.markdown("---")

            # 응답 평가 및 레이더 차트 표시
            st.subheader("응답 평가")
            
            # GPT-4o의 평가
            st.markdown("### GPT-4o의 평가")
            gpt4_scores_gpt = evaluate_response(gpt4_response)
            gpt4_scores_claude = evaluate_response(claude_response)
            gpt4_scores_gemini = evaluate_response(gemini_response)
            
            fig_gpt = go.Figure()
            
            # 각 모델별 색상 정의 (rgba 형식 사용)
            colors = {
                'GPT-4o': 'rgba(255, 107, 107, 1)',  # 선명한 빨간색
                'Claude-3.5': 'rgba(255, 193, 69, 1)',  # 선명한 주황/노란색
                'Gemini Pro': 'rgba(69, 183, 209, 1)'  # 하늘색
            }
            
            colors_fill = {
                'GPT-4o': 'rgba(255, 107, 107, 0.3)',  # 빨간색 투명
                'Claude-3.5': 'rgba(255, 193, 69, 0.3)',  # 주황/노란색 투명
                'Gemini Pro': 'rgba(69, 183, 209, 0.3)'  # 하늘색 투명
            }
            
            fig_gpt.add_trace(go.Scatterpolar(
                r=gpt4_scores_gpt, 
                theta=categories, 
                fill='toself', 
                name='GPT-4o', 
                line=dict(color=colors['GPT-4o']), 
                fillcolor=colors_fill['GPT-4o']
            ))
            fig_gpt.add_trace(go.Scatterpolar(
                r=gpt4_scores_claude, 
                theta=categories, 
                fill='toself', 
                name='Claude-3.5', 
                line=dict(color=colors['Claude-3.5']), 
                fillcolor=colors_fill['Claude-3.5']
            ))
            fig_gpt.add_trace(go.Scatterpolar(
                r=gpt4_scores_gemini, 
                theta=categories, 
                fill='toself', 
                name='Gemini Pro', 
                line=dict(color=colors['Gemini Pro']), 
                fillcolor=colors_fill['Gemini Pro']
            ))
            
            fig_gpt.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=True,
                title="GPT-4o의 평가 결과"
            )
            
            st.plotly_chart(fig_gpt, use_container_width=True)
            
            # GPT-4o의 총평 추가
            st.markdown("#### GPT-4o의 총평")
            responses = {
                'prompt': prompt,
                'gpt4': gpt4_response,
                'claude': claude_response,
                'gemini': gemini_response
            }
            with st.spinner("GPT-4o가 평가 중..."):
                gpt4_evaluation = get_summary_evaluation("GPT-4o", responses)
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
                {gpt4_evaluation}
                </div>
                """, unsafe_allow_html=True)

            # Claude의 평가
            st.markdown("### Claude-3.5의 평가")
            claude_scores_gpt = evaluate_response(gpt4_response)
            claude_scores_claude = evaluate_response(claude_response)
            claude_scores_gemini = evaluate_response(gemini_response)
            
            fig_claude = go.Figure()
            
            # 각 모델별 색상 정의 (rgba 형식 사용)
            colors = {
                'GPT-4o': 'rgba(255, 107, 107, 1)',  # 빨간색
                'Claude-3.5': 'rgba(78, 205, 196, 1)',  # 청록색
                'Gemini Pro': 'rgba(69, 183, 209, 1)'  # 하늘색
            }
            
            colors_fill = {
                'GPT-4o': 'rgba(255, 107, 107, 0.3)',  # 빨간색 투명
                'Claude-3.5': 'rgba(78, 205, 196, 0.3)',  # 청록색 투명
                'Gemini Pro': 'rgba(69, 183, 209, 0.3)'  # 하늘색 투명
            }
            
            fig_claude.add_trace(go.Scatterpolar(
                r=claude_scores_gpt, 
                theta=categories, 
                fill='toself', 
                name='GPT-4o', 
                line=dict(color=colors['GPT-4o']), 
                fillcolor=colors_fill['GPT-4o']
            ))
            fig_claude.add_trace(go.Scatterpolar(
                r=claude_scores_claude, 
                theta=categories, 
                fill='toself', 
                name='Claude-3.5', 
                line=dict(color=colors['Claude-3.5']), 
                fillcolor=colors_fill['Claude-3.5']
            ))
            fig_claude.add_trace(go.Scatterpolar(
                r=claude_scores_gemini, 
                theta=categories, 
                fill='toself', 
                name='Gemini Pro', 
                line=dict(color=colors['Gemini Pro']), 
                fillcolor=colors_fill['Gemini Pro']
            ))
            
            fig_claude.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=True,
                title="Claude-3.5의 평가 결과"
            )
            
            st.plotly_chart(fig_claude, use_container_width=True)
            
            # Claude의 총평 추가
            st.markdown("#### Claude-3.5의 총평")
            with st.spinner("Claude-3.5가 평가 중..."):
                claude_evaluation = get_summary_evaluation("Claude-3.5", responses)
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
                {claude_evaluation}
                </div>
                """, unsafe_allow_html=True)

            # Gemini의 평가
            st.markdown("### Gemini Pro의 평가")
            gemini_scores_gpt = evaluate_response(gpt4_response)
            gemini_scores_claude = evaluate_response(claude_response)
            gemini_scores_gemini = evaluate_response(gemini_response)
            
            fig_gemini = go.Figure()
            
            # 각 모델별 색상 정의 (rgba 형식 사용)
            colors = {
                'GPT-4o': 'rgba(255, 107, 107, 1)',  # 빨간색
                'Claude-3.5': 'rgba(78, 205, 196, 1)',  # 청록색
                'Gemini Pro': 'rgba(69, 183, 209, 1)'  # 하늘색
            }
            
            colors_fill = {
                'GPT-4o': 'rgba(255, 107, 107, 0.3)',  # 빨간색 투명
                'Claude-3.5': 'rgba(78, 205, 196, 0.3)',  # 청록색 투명
                'Gemini Pro': 'rgba(69, 183, 209, 0.3)'  # 하늘색 투명
            }
            
            fig_gemini.add_trace(go.Scatterpolar(
                r=gemini_scores_gpt, 
                theta=categories, 
                fill='toself', 
                name='GPT-4o', 
                line=dict(color=colors['GPT-4o']), 
                fillcolor=colors_fill['GPT-4o']
            ))
            fig_gemini.add_trace(go.Scatterpolar(
                r=gemini_scores_claude, 
                theta=categories, 
                fill='toself', 
                name='Claude-3.5', 
                line=dict(color=colors['Claude-3.5']), 
                fillcolor=colors_fill['Claude-3.5']
            ))
            fig_gemini.add_trace(go.Scatterpolar(
                r=gemini_scores_gemini, 
                theta=categories, 
                fill='toself', 
                name='Gemini Pro', 
                line=dict(color=colors['Gemini Pro']), 
                fillcolor=colors_fill['Gemini Pro']
            ))
            
            fig_gemini.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                showlegend=True,
                title="Gemini Pro의 평가 결과"
            )
            
            st.plotly_chart(fig_gemini, use_container_width=True)
            
            # Gemini의 총평 추가
            st.markdown("#### Gemini Pro의 총평")
            with st.spinner("Gemini Pro가 평가 중..."):
                gemini_evaluation = get_summary_evaluation("Gemini Pro", responses)
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
                {gemini_evaluation}
                </div>
                """, unsafe_allow_html=True)

            # 평가 기준 설명
            st.subheader("평가 기준 설명")
            st.markdown("""
            - **정확성**: 제공된 정보의 사실적 정확도와 신뢰���
            - **창의성**: 독창적이고 혁신적인 아이디어 제시 능력
            - **논리성**: 논리적 구조와 일관성 있는 설명
            - **완성도**: 응의 포괄성과 완결성
            - **유용성**: 실제 적용 가능성과 실용적 가치
            """)
        else:
            st.warning("프롬프트를 입력해주세요!")

if __name__ == "__main__":
    main() 