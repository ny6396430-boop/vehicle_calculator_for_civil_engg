from ultralytics import YOLO
import cv2
import argparse
import time
import os
from tqdm import tqdm

VEHICLE_CLASSES = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'motorcycle': 'motorcycle',
    'bicycle': 'motorcycle'
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Path to input video')
    p.add_argument('--output', default='output.mp4', help='Path to save annotated video')
    p.add_argument('--model', default='yolov8n.pt', help='YOLOv8 model path or name')
    p.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    p.add_argument('--line', type=float, default=0.5, help='Vertical line position (relative, 0-1) for counting')
    p.add_argument('--resize', type=int, default=1280, help='Max width for processing (keeps aspect ratio)')
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(args.source)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width > args.resize:
        scale = args.resize / width
        width = args.resize
        height = int(height * scale)
    else:
        scale = 1.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    counts = {k: 0 for k in set(VEHICLE_CLASSES.values())}
    seen_ids = {k: set() for k in counts.keys()}

    line_y = int(height * args.line)

    print('Starting tracking & counting...')
    start = time.time()

    results = model.track(source=args.source, persist=True, conf=args.conf, tracker='bytetrack.yaml')

    for res in tqdm(results):
        frame = res.orig_img
        if frame is None:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))

        if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes.data) > 0:
            for row in res.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls, tid = row
                cls = int(cls)
                name = model.names.get(cls, str(cls))
                if name in VEHICLE_CLASSES:
                    vtype = VEHICLE_CLASSES[name]
                    tid = int(tid) if not (tid is None) else None
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    if tid is not None:
                        if tid not in seen_ids[vtype]:
                            if cy < line_y:
                                counts[vtype] += 1
                                seen_ids[vtype].add(tid)
                    else:
                        if cy < line_y:
                            counts[vtype] += 1

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{vtype} {int(conf*100)}%"
                    cv2.putText(frame, label, (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.line(frame, (0, line_y), (width, line_y), (0,0,255), 2)
        status_text = '  '.join([f"{k}: {v}" for k,v in counts.items()])
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        out.write(frame)

    out.release()
    cap.release()
    elapsed = time.time() - start
    print('\nDone. Output saved to', args.output)
    print('Elapsed (s):', round(elapsed,2))
    print('Final counts:', counts)

if __name__ == '__main__':
    main()
