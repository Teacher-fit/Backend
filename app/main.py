from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
import openai
import os
import json
from dotenv import load_dotenv

app = FastAPI()

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
def load_model_parameters(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        parameters = json.load(file)
    return parameters


# 파일 경로에 따라 교과서 내용을 로드하는 함수
def load_textbook_content_from_input(
    main_chapter, sub_chapter, small_chapter, base_dir="./data/textbook/"
):
    file_name = f"{main_chapter}_{sub_chapter}_{small_chapter}.txt"
    file_path = os.path.join(base_dir, file_name)

    # 파일이 존재하는지 확인하고 내용을 읽어오기
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    else:
        return f"파일을 찾을 수 없습니다: {file_path}"


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
[단원명]

## 준비물
[준비물이 필요한 경우 제시해주세요.]

## 수업 방식
[수업 방식]

## 에듀테크 제품
[제품명 및 추천 이유/활용 방안]

## 수업 목표
- 목표

## 단계별 학습활동
아래 표에 각 단계별로 구체적인 학습활동, 활동의 목적, 진행 방법, 학생들의 역할, 그리고 수업형태를 상세히 작성해 주세요. 추천한 에듀테크 제품을 활동 중 적절하게 사용할 수 있도록 창의적인 답변을 주세요.

| **단계** | **학습활동** | **수업형태** |
|---|---|---|
| 도입 | [예: 학생들의 관심을 끌기 위한 동기 부여 활동] | [수업형태] |
| 전개1 | [예: 에듀테크 도구를 활용한 핵심 개념 학습] | [수업형태] |
| 전개2 | [예: 그룹 활동을 통한 문제 해결] | [수업형태] |
| 마무리 | [예: 학습 내용 정리 및 피드백] | [수업형태] |


## 활용효과
- 효과
"""
    try:
        # ChatCompletion 생성 요청
        response = openai.ChatCompletion.create(
            model=model_parameters["model_name"],  # 모델 이름
            messages=[
                {
                    "role": "system",
                    "content": "당신은 중학생 대상 교육 활동 지도가이자 에듀테크 컨텐츠 추천 전문가입니다.",
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
            # logit_bias=model_parameters["logit_bias"]  # 필요 시 사용
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"


# 데이터 받아서 GPT API를 호출하고 응답을 반환
@app.post("/edu-recommend", response_model=ResponseData)
async def create_response(data: RequestData = Body(...)):
    # 모델 파라미터 로드
    model_parameters = load_model_parameters("model.json")

    # 교과서 내용 로드
    textbook_content = load_textbook_content_from_input(
        data.main_chapter, data.sub_chapter, data.small_chapter
    )

    # GPT API를 호출하여 학습 활동 추천
    activities = recommend_learning_activities_in_korean(
        textbook_content, data, model_parameters
    )

    # dict 형식으로 반환
    response_data = {"content": activities}

    return response_data
