# CVAT Annotation Workflow

This note covers the local CVAT setup for manually correcting the
`small_meeting_xiding_cvat` COCO keypoint dataset.

## Current State

- CVAT source directory: `/home/hebo/code/cvat`
- Dataset directory:
  `/home/hebo/code/MeetEye/fine-tune/datasets/small_meeting_xiding_cvat`
- Import packages:
  - `small_meeting_xiding_cvat_train_coco_keypoints.zip`
  - `small_meeting_xiding_cvat_test_coco_keypoints.zip`

The source annotations are in COCO keypoint JSON:

- `train/annotations_coco_keypoints.json`
- `test/annotations_coco_keypoints.json`

The generated ZIP files use the CVAT/DATUMARO COCO Keypoints layout:

```text
annotations/person_keypoints_<split>.json
images/<split>/<image files>
```

## Start CVAT

Run these commands in a normal terminal on the host. The current shell user
cannot access the Docker daemon directly, so `sudo` is required unless Docker
group permissions are fixed.

```bash
cd /home/hebo/code/cvat
sudo docker compose up -d
```

Wait until the stack is healthy:

```bash
sudo docker compose ps
```

Create the first admin user:

```bash
sudo docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Then open:

```text
http://localhost:8080
```

If accessing CVAT from another machine, start it with `CVAT_HOST` set to the
server IP or DNS name:

```bash
cd /home/hebo/code/cvat
CVAT_HOST=<server-ip-or-domain> sudo -E docker compose up -d
```

## Import For Labeling

In CVAT:

1. Create a task or project.
2. Choose format `COCO Keypoints 1.0`.
3. Upload one of:
   - `fine-tune/datasets/small_meeting_xiding_cvat/small_meeting_xiding_cvat_train_coco_keypoints.zip`
   - `fine-tune/datasets/small_meeting_xiding_cvat/small_meeting_xiding_cvat_test_coco_keypoints.zip`
4. Keep the label as `person` with the COCO 17-keypoint skeleton.

The dataset sizes are:

- train: 1107 images, 1200 annotations
- test: 276 images, 300 annotations

## Export Back To COCO

After annotation:

1. Open the task/project.
2. Use `Export dataset`.
3. Select `COCO Keypoints 1.0`.
4. Enable image export only if a full image+annotation archive is needed.

CVAT exports a ZIP with:

```text
annotations/person_keypoints_<subset>.json
images/<subset>/
```

Use the exported `person_keypoints_*.json` as the corrected COCO annotation
source. If the export uses a different subset name, keep the JSON but normalize
image paths before feeding it to downstream training scripts.

## Stop CVAT

Stop services without deleting volumes:

```bash
cd /home/hebo/code/cvat
sudo docker compose down
```

Remove CVAT volumes only when the database and uploaded data are no longer
needed:

```bash
cd /home/hebo/code/cvat
sudo docker compose down -v
```
