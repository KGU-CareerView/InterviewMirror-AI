import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def test():
    if not api_key:
        print("❌ 에러: .env 파일에 GEMINI_API_KEY가 없습니다.")
        return
    
    client = genai.Client(api_key=api_key)
    
    print("--- [1] 내 API 키로 접근 가능한 전체 모델 목록 ---")
    available_models = []
    try:
        # 속성 필터링 없이 이름만 싹 다 가져옵니다.
        for m in client.models.list():
            print(f"발견된 모델: {m.name}")
            available_models.append(m.name)
    except Exception as e:
        print(f"목록 조회 실패: {e}")

    print("\n--- [2] API 호출 테스트 시도 ---")
    # 목록에 'models/gemini-1.5-flash'가 있다면 그대로 쓰고, 
    # 없으면 목록 중 가장 첫 번째 모델을 자동으로 선택해서 시도합니다.
    target_model = "models/gemini-1.5-flash" 
    if available_models and target_model not in available_models:
        target_model = available_models[0]
        print(f"💡 {target_model}로 우회하여 시도합니다.")

    try:
        response = client.models.generate_content(
            model=target_model,
            contents="성공이라고 대답해줘."
        )
        print("--- 결과 ---")
        print(response.text)
    except Exception as e:
        print(f"❌ 최종 시도 에러: {e}")

if __name__ == "__main__":
    test()