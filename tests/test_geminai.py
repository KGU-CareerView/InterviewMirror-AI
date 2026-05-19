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
    
    # 1. 사용 가능한 모델을 직접 출력해서 확인 (디버깅용)
    print("--- 사용 가능한 모델 목록 ---")
    try:
        for m in client.models.list():
            if 'generateContent' in m.supported_generation_methods:
                print(f"ID: {m.name}")
    except Exception as e:
        print(f"모델 목록 조회 실패: {e}")

    # 2. 가장 확실한 이름으로 테스트
    # 최신 SDK에서는 'gemini-1.5-flash' 혹은 'gemini-2.0-flash'라고만 써야 합니다.
    print("\n--- API 호출 테스트 시작 ---")
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", # 또는 "gemini-2.0-flash"
            contents="연결 성공 여부를 확인합니다. '성공'이라고 대답해줘."
        )
        print("--- 결과 ---")
        print(response.text)
    except Exception as e:
        print(f"❌ 또 에러 발생: {e}")

if __name__ == "__main__":
    test()