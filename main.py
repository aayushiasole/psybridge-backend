from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mbti_model import predict_mbti

app = FastAPI(title="MBTI Predictor API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with your frontend URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "MBTI Predictor Backend is Running!"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    answers = data.get("answers", [])

    if not isinstance(answers, list) or len(answers) != 60:
        return {"error": "Invalid input: Expected a list of 60 numeric answers."}

    try:
        mbti_type = predict_mbti(answers)
        return {"mbti_type": mbti_type}
    except Exception as e:
        return {"error": str(e)}
