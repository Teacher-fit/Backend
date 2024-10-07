from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# .env 파일에서 API 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# 요청에 대한 데이터 모델 정의
class RequestData(BaseModel):
    grade: int  # 중1, 중2, 중3, 전체 등의 학년 선택
    main_chapter: int  # 대단원
    sub_chapter: int  # 중단원
    small_chapter: int  # 소단원
    requirement: Optional[str]  # 요청사항 (선택 사항)
    use_edutech_tool: bool  # 에듀테크 도구 활용 여부 (체크박스)


# 응답 모델 정의
class ResponseData(BaseModel):
    content: str


# 모델 파라미터 로드 함수
def load_model_parameters(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        parameters = json.load(file)
    return parameters


# 파일 경로에 따라 교과서 내용을 로드하는 함수
def load_textbook_content_from_input(
    main_chapter, sub_chapter, small_chapter, base_dir="./data/textbook/"
):
    file_name = f"{main_chapter+1}_{sub_chapter+1}_{small_chapter+1}.txt"
    file_path = os.path.join(base_dir, file_name)

    # 파일이 존재하는지 확인하고 내용을 읽어오기
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    else:
        raise FileNotFoundError(f"File {file_name} not found at {base_dir}")


# GPT API를 사용하여 학습 활동을 추천하는 함수
def recommend_learning_activities_in_korean(
    textbook_content, data: RequestData, model_parameters
):
    prompt = f"""
다음은 응답의 기본이 될 교과서의 내용입니다:
{textbook_content}

다음은 사용자의 요청 정보입니다:
- 학년: 중학교 {data.grade}학년
- 과목: 도덕
- 요청사항: {data.requirement if data.requirement else '없음'}

**요청 사항이 있는 경우, 반드시 요청 사항을 우선적으로 반영하세요**

교사의 수업 중 다양한 에듀테크 도구들을 수업에 활용할 수 있도록 장려하려고 합니다.
교과서의 학습 활동 내용과 사용자 요청을 바탕으로, 중학생들의 수업 시간에 적용할 수 있는 에듀테크 기반의 학습 활동을 제안해 주세요.

**응답은 다음의 마크다운 형식을 따라 주세요:**

## 교과
[교과명]

## 대상
[대상 학년]

## 단원
대단원: {data.main_chapter+1}, 중단원: {data.sub_chapter+1}, 소단원: {data.small_chapter+1}, 단원명 : {textbook_content}의 첫번째 줄 (숫자 제외).
**교과서 내용을 출력하지 말고 단원 정보만 출력해**
[출력 ex) 대단원 2, 중단원 1, 소단원 3 : 우리 안에 있는 다문화의 모습은 무엇인가?]

## 준비물
[준비물이 필요한 경우 제시해주세요.]

## 수업 방식
[수업 방식]

## 에듀테크 제품

[제품명: 추천 에듀테크 도구(앱, 프로그램 등) 및 해당 도구를 추천하는 이유와 구체적 활용 방안]

## 교사들을 위한 에듀테크 도구 사용법

- **사용법**: 교사들이 에듀테크 도구를 수업 중 어떻게 사용할 수 있는지 간단하고 명료하게 설명해 주세요. 설치 방법, 기본 기능, 수업 중 적용 방법 등을 포함하여 교사들이 도구를 효과적으로 활용할 수 있도록 안내해 주세요.

## 수업 목표
- 학습 목표: 이 수업을 통해 학생들이 달성해야 할 구체적인 목표

## 단계별 학습활동
아래 표에 각 단계별로 구체적인 학습활동, 활동의 목적, 진행 방법, 학생들의 역할, 그리고 수업형태를 상세히 작성해 줘. 추천한 에듀테크 제품을 활동 중 적절하게 사용할 수 있도록 창의적인 답변을 줘.
학습 활동은 도입, 전개, 마무리 각각 **200자 분량**으로 교사가 따라할 수 있도록 구체적인 활동을 제시해줘.

| **단계** | **학습활동** | **수업형태** |
|---|---|---|
| 도입 | [ex. 에듀테크 도구를 활용하여 학생들의 흥미를 유도하고 수업의 배경 지식을 제공합니다.] | [수업형태] |
| 전개1 | [ex. 에듀테크 도구를 활용하여 핵심 개념을 학습하고, 직관적인 시각 자료를 통해 이해를 돕습니다.] | [수업형태] |
| 전개2 | [ex. 그룹 활동을 통해 문제를 해결하고, 학습 내용을 실제 생활에 적용할 수 있는 상황을 제공합니다.] | [수업형태] |
| 마무리 | [ex. 학생들이 학습 내용을 요약하고, 에듀테크 도구를 통해 피드백을 받습니다.] | [수업형태] |


## 활용효과
- 효과

## 지도상 유의사항
ex) 
- **기술 격차**: 학생들이 각기 다른 디지털 리터러시 수준을 가지고 있을 수 있으므로, 모든 학생들이 도구를 사용할 수 있도록 초기 단계에서 충분한 지원을 제공하세요.
- **학생 참여도**: 에듀테크 도구를 사용하는 활동은 흥미를 끌 수 있지만, 도구 사용에 집중하여 학습 목표를 소홀히 하지 않도록 유도하는 것이 중요합니다.
- **윤리적 문제**: 디지털 도구를 사용할 때, 개인정보 보호와 온라인에서의 윤리적인 행동에 대해 명확히 가르쳐야 합니다.
"""
    try:
        # ChatCompletion 생성 요청
        response = openai.ChatCompletion.create(
            model=model_parameters["model_name"],  # 모델 이름
            messages=[
                {
                    "role": "system",
                    "content": "당신은 중학생 대상 교수 설계 전문가이다. 요청을 바탕으로 교수학습 지도안을 설계해라.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=model_parameters["max_tokens"],  # 최대 토큰 수
            temperature=model_parameters["temperature"],  # 응답의 무작위성 제어
            top_p=model_parameters["top_p"],  # 확률 분포 제어
            n=model_parameters["n"],  # 생성할 응답의 수
            presence_penalty=model_parameters[
                "presence_penalty"
            ],  # 새로운 주제 생성 유도
            frequency_penalty=model_parameters[
                "frequency_penalty"
            ],  # 반복 내용 생성 방지
            request_timeout=model_parameters[
                "request_timeout"
            ],  # 요청 시간 제한 (없으면 None)
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"GPT API 호출 중 에러 발생: {e}")
        return f"Error: {str(e)}"


# 데이터 받아서 GPT API를 호출하고 응답을 반환
@app.post("/edu-recommend", response_model=ResponseData)
async def create_response(data: RequestData = Body(...)):
    try:
        model_parameters = load_model_parameters("model.json")
        textbook_content = load_textbook_content_from_input(
            data.main_chapter, data.sub_chapter, data.small_chapter
        )
        activities = recommend_learning_activities_in_korean(
            textbook_content, data, model_parameters
        )
        response_data = {"content": activities}
        return response_data
    except FileNotFoundError as e:
        logging.error(f"파일 로드 에러: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"예기치 못한 에러 발생: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
