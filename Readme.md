# PANDA Prostate Tumor Segmentation 
## Results 
![](Images/Tumor_Results.png)
## TL;DR
- **Dataset:** [PANDA (Prostate cANcer graDe Assessment)](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment)  
- **Goal:** segment **cancer** pixels vs non-cancer  
- **Core functions to look at:**
  - `find_one_tumor_example()` – picks one slide that actually has tumor
  - `tumor_bool_from_gray(mask_gray)` – converts raw mask labels → **binary cancer mask**
  - `tissue_mask_rgb(img_rgb)` – rough tissue detector for patch filtering (not labels)
  - **Patching:** code block where we tile with `slide.read_region(...)` and aligned `mask.read_region(...)`
  - **Dataset:** `class PatchSet(Dataset)` – loads saved patches for training
  - **Training:** `run_epoch(...)` – one pass over dataloader
  - **Inference:** we build a `pred_canvas` (probability map) over the slide
  - **Heatmap & Quant:** `slide_quantification(...)` (or the inline quant code) does area/percent

---

## Folder layout (Kaggle Working)
```
/kaggle/working/
  aria_seg_L0/               # or simple_demo/, depends which snippet you ran
    <SLIDE_ID>/
      img/                   # saved image patches (PNG)
      msk/                   # saved mask patches (PNG, binary cancer mask)
      train.csv              # manifest for training
      val.csv                # manifest for validation
```
Patches names contains coords like `SID_x1234_y5678.png` so its easy to trace back.

---

## 1) Dataset & assumptions
- We use **`train_images/`** and **`train_label_masks/`** from PANDA.  
- Masks differ by provider. Our helper **`tumor_bool_from_gray`** maps them to **binary cancer**:
  - Radboud: tumor classes = **{3,4,5}**
  - Karolinska: tumor class = **2**  
Everything else (0,1,2 benign etc.) becomes non-cancer.

> If your sample looks all green before, it was benign class being visualized. Binary mapping fixes that.

---

## 2) Zoomed‑out slide + mask
We first pick a slide with tumor via **`find_one_tumor_example()`**, then:  
- `slide.get_thumbnail(...)` to show the WSI quickly  
- `mask.get_thumbnail(...)` with the same size, convert to numpy (`mask_gray`)  
- Overlay for sanity

This makes sure image↔mask alignment is right before we go deeper.

---

## 3) Patching (512×512)
We read **aligned tiles** at a chosen **pyramid level** using:
- `slide.read_region((x0, y0), LEVEL, (PATCH_SIZE, PATCH_SIZE))`
- `mask.read_region((mx0, my0), LEVEL_M, (PATCH_SIZE, PATCH_SIZE))`

Coordinates are mapped with:
- `rx, ry = W0_m/W0_s, H0_m/H0_s` (ratio slide L0 to mask L0)  
So mask patch lines up with image patch.  
We store PNGs and a **manifest CSV** for training/val.

> Tissue filter uses `tissue_mask_rgb(...)`. It **only** avoids glass/blank, it’s not a labeler.

---

## 4) Training (DeepLabV3)
- Model: **DeepLabV3-ResNet50** (binary head)  
- Loss: **BCE + Dice** (good for class imbalance)  
- Data: `PatchSet` yields `(image, mask)` tensors in `[0,1]`  
- Loop: `run_epoch(dataloader, train=True/False)` prints total loss + approx Dice

You can train at **LEVEL=0** (full-res) or lighter levels. L0 is slower but sharper.

---

## 5) Inference → Probability map → Heatmap
- We sweep the slide grid again and call `model(tile)["out"]`  
- Accumulate into `pred_canvas` (probabilities ∈ [0,1])  
- **Heatmap** is just a colormap over probabilities (e.g., JET).  
- **Threshold**: start at 0.5, but better pick it from validation (we also provide a helper for threshold/F1 search).

---

## 6) Tumor quantification
From `pred_canvas`:
- **Binary prediction** = `pred > THRESH`
- **Tissue mask** = `tissue_mask_rgb(...)` (so glass isn’t counted)
- **Tumor burden (% of tissue)** = tumor_pixels / tissue_pixels × 100
- If slide properties have `openslide.mpp-x/y`, we compute **tumor area mm²**.  
- Optionally do small-object removal and hole-filling to make counts realistic.

Function: **`slide_quantification(pred_canvas, slide, level_eval, thr=BEST_T)`** returns a dict with:
```
{ tumor_px, tissue_px, tumor_percent_of_tissue, tumor_mm2, tissue_mm2, num_regions, mask_clean }
```

---

## 7) Patch‑wise views
For quick QA we show rows of 3–5 panels:
- Original **image patch**
- **Predicted mask** (binary)
- **Overlay** (image + red mask)
- **Ground truth** patch
- **Heatmap** (per‑patch probabilities)

This helps to eyeball false positives/negatives right away.

---

## 8) (Optional) Slide‑level grading with CNN+MIL
When you want **ISUP grade** or simple **cancer present/absent** per slide (no pixel masks needed):  
- Use your heatmap to pick **K tumor‑likely patches**  
- Extract features with a CNN and combine via **Attention MIL** (class `MILNet`)  
- Function points:
  - `select_tile_coords_for_slide(...)` – choose top‑K coords
  - `read_bag(...)` – read a bag of K patches
  - `MILNet.forward(bag)` – returns `(logits, attention_weights, feats)`

**Segmentation + MIL** combo works great: segmentation finds “where”, MIL decides slide label.

---

## How to run (short recipe)
1. Open Kaggle notebook (GPU).  
2. Run the single‑snippet L0 pipeline (it creates `img/`, `msk/`, csvs, and trains).  
3. Run the **heatmap & quant** cell to get tumor burden and colored heatmaps.  
4. Optional: run the **MIL snippet** if you want slide ISUP grading demo.

---

## Tips & gotchas
- **Levels/scale:** LEVEL=0 has best detail; otherwise LEVEL=1 is faster.  
- **Threshold:** don’t stick to 0.5; fit on validation to maximize Dice/F1.  
- **Class mapping:** always check `np.unique(mask_gray)` before training.  
- **Imbalance:** add Dice loss, sample more positives, or use hard‑negative mining later.  
- **Stain variance:** add HSV/brightness jitter aug if you expand beyond one site.  
- **Memory:** reduce BATCH_SIZE or tile stride if you run out of VRAM.

---

## FAQ
**Q: Are we detecting tumor or cancer?**  
A: In this repo, “tumor” means **cancer** pixels. Benign tissue is negative. See `tumor_bool_from_gray(...)` mapping.

**Q: Is the heatmap a model?**  
A: No. The **probability map** comes from the model; heatmap is just a **colormap** on those probabilities.

**Q: Why some slides have no cancer?**  
A: PANDA includes benign slides. If `np.unique(mask_gray)` has no tumor classes, it’s benign—don’t worry, it’s expected.

---

## Credits
- PANDA dataset authors & institutions.  
- OpenSlide, PyTorch, Torchvision.  
- Thanks to open‑source community for ideas, we adapted parts in a very pragmatic way.

---

## Changelog (mini)
- v0.1 — first cut: L0 patching, DeepLabV3, heatmap & quant, simple MIL demo
- v0.2 — docs cleanup (still some grammar, cuz I’m not a robot lol)


