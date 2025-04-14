# AI Visitor Identification with Analytics

## Features
- Analyze a pre-uploaded video (`sample_video.webm`)
- Count unique visitors using YOLOv8 + Deep SORT
- View analytics summary as JSON
- Deployable via Render.com

## Endpoints
- `GET /process-sample` → Process the sample video and log analytics
- `GET /analytics` → Return summary (unique visitor count)

## To Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then go to: `http://localhost:8000/docs`