from __future__ import annotations
from typing import Dict, List, Optional
from langchain_core.documents import Document


INSTRUCTION_RAG = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
An anomaly detection model has analyzed this image and provided the following information:
{ad_info}

Here is the domain knowledge about known defect types for this product:
{domain_knowledge}

Use the anomaly detection results and domain knowledge along with your visual analysis to answer the questions.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# -- 위 prompt 내용 --

# 너는 이미지로 제품을 검사하는 산업 검사관이야.                                                       
# 쿼리 이미지에 결함이 있는지 판단하고 질문에 답해.                                                  
                                                                                                    
# 이상 탐지 모델이 이 이미지를 분석한 결과야:                                                          
# {ad_info}                                                                                            
                                                                                                    
# 이 제품의 알려진 결함 유형에 대한 도메인 지식이야:
# {domain_knowledge}

# 이상 탐지 결과와 도메인 지식, 그리고 네 시각 분석을 활용해서 답해.
# 선택지의 알파벳으로 바로 답해!

# 답변은 이렇게 출력해:
# 1. Answer: B.
# 2. Answer: B.
# 3. Answer: A.



REPORT_PROMPT_RAG = '''당신은 제조 품질관리 수석 검사관입니다.
제품 카테고리: {category}

AD 사전 분석:
{ad_info}

도메인 지식:
{domain_knowledge}

판정 규칙:
- AD decision=ANOMALY 이면 is_anomaly=true로 고정하세요.
- AD decision=NORMAL 이면 is_anomaly=false로 고정하세요.
- AD decision=REVIEW_NEEDED 이면 AD 모델의 판정은 무시하고 이미지 근거로 is_anomaly를 판단하세요.

RAG 사용:
- 도메인 지식은 결함명/원인 설명 보조에만 사용하세요.
- 이미지 근거 없이 도메인 지식 문구를 단정적으로 복사하지 마세요.

리포트 규칙:
- 최종 판정이 이상이면 원인 분석과 시정/예방 조치를 구체적으로 작성하세요.
- 최종 판정이 정상이면 정상 근거를 간결하게 작성하세요.
- 보이지 않는 결함/원인/위치는 추측하지 마세요.

출력 규칙:
- JSON만 출력하세요.
- 문자열은 한국어로 작성하세요.
- severity, risk_level은 low/medium/high/none 중 하나.
- confidence는 0.0~1.0 숫자.
- is_anomaly=false이면 anomaly_type/severity/location/possible_cause/risk_level은 모두 "none".

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "is_anomaly": true or false,
  "report": {{
    "anomaly_type": "specific defect type or none",
    "severity": "low/medium/high/none",
    "location": "defect location or none",
    "description": "근거 중심의 상세 설명(한국어)",
    "possible_cause": "가장 가능성 높은 원인 또는 none",
    "confidence": 0.0 to 1.0,
    "recommendation": "구체적 시정/예방 조치(한국어)"
  }},
  "summary": {{
    "summary": "정확히 3문장: 최종 판정 + 핵심 근거/긴급도 + 구체적 시정/예방조치(한국어)",
    "risk_level": "low/medium/high/none"
  }}
}}'''


def report_prompt_rag(
    category: str,
    domain_knowledge: str = "",
    ad_info: str = "",
) -> str:
    """Build a RAG-augmented report generation prompt.

    Args:
        category: Product category string.
        domain_knowledge: Formatted domain knowledge from retriever.format_context().
        ad_info: Formatted AD model output string.

    Returns:
        Filled report prompt string.
    """
    return REPORT_PROMPT_RAG.format(
        category=category,
        ad_info=ad_info or "No anomaly detection information available.",
        domain_knowledge=domain_knowledge or "No relevant domain knowledge found.",
    )


def rag_prompt(
    ad_info: str = "",
    domain_knowledge: str = "",
) -> str:
    """Build a complete RAG-augmented instruction prompt.

    Args:
        ad_info: Formatted AD model output string.
        domain_knowledge: Formatted domain knowledge from retriever.format_context().

    Returns:
        Filled instruction prompt string.
    """
    return INSTRUCTION_RAG.format(
        ad_info=ad_info or "No anomaly detection information available.",
        domain_knowledge=domain_knowledge or "No relevant domain knowledge found.",
    )
