import os
from dotenv import load_dotenv
from google import genai

# .env 파일에서 API 키를 읽어옵니다.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def test():
    if not api_key:
        print("❌ 에러: .env 파일에 GEMINI_API_KEY가 없습니다.")
        return
    
    # 최신 SDK 클라이언트 생성
    client = genai.Client(api_key=api_key)
    
    try:
     # 수정 전: model="gemini-1.5-flash"
# 수정 후:
response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents="연결 성공 여부를 확인합니다. '성공'이라고 대답해줘."
)
        print("--- 결과 ---")
        print(response.text)
    except Exception as e:
        print(f"❌ API 호출 중 에러 발생: {e}")

if __name__ == "__main__":
    test()